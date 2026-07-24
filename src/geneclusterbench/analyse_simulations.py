#!/usr/bin/env python3

import os
import glob
import argparse
import re
import itertools
import subprocess
import tempfile
import shutil

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA


# List of the 30 simulation seeds. Used to reliably pull the seed
# out of a file path even if other numeric folders appear on the way.
SEEDS = [
    34, 53144, 40547, 60207, 21708, 31001, 54634, 29492, 6032, 30354,
    5319, 46118, 1681, 27347, 14928, 14557, 62092, 49444, 25172, 25913,
    31375, 13478, 14720, 1274, 11998, 5455, 56065, 35787, 28734, 1894,
]
SEED_STRS = {str(s) for s in SEEDS}


# ---------------------------------------------------------
# FILE HANDLING
# ---------------------------------------------------------

def extract_seed(path):
    """
    Pull the seed out of a path by matching against the known SEEDS list.
    Looks at every path component (folder names and filename), returns
    the first one that matches a known seed. Returns None if not found.
    """
    parts = re.split(r"[\\/]", path)
    for part in parts:
        for token in re.findall(r"\d+", part):
            if token in SEED_STRS:
                return int(token)
    return None


def find_files(folder):
    return glob.glob(
        os.path.join(folder, "**/*_presence_absence.csv"),
        recursive=True
    )


def load_pa(path):
    return pd.read_csv(path, sep="\t").iloc[:, 1:]

def find_gffs(folder):

    return glob.glob(
        os.path.join(folder, "**/*.gff"),
        recursive=True
    )


def read_gff(gff_file):

    rows = []

    with open(gff_file) as fh:

        for line in fh:

            if line.startswith("#"):
                continue

            fields = line.rstrip().split("\t")

            if len(fields) != 9:
                continue

            feature = fields[2]

            if feature not in ["CDS", "gene"]:
                continue

            rows.append({
                "seqid": fields[0],
                "start": int(fields[3]),
                "end": int(fields[4]),
                "strand": fields[6],
                "feature": feature,
                "attributes": fields[8]
            })

    return pd.DataFrame(rows)


def read_gff_sequence(gff_file):
    """
    Prokka/GFF3 files often carry the genome sequence itself after a
    '##FASTA' directive at the end of the file. Extract and concatenate
    it (all contigs) into one string. Returns "" if no FASTA block found.
    """
    seq_lines = []
    in_fasta = False

    with open(gff_file) as fh:
        for line in fh:
            if line.startswith("##FASTA"):
                in_fasta = True
                continue
            if in_fasta:
                if line.startswith(">"):
                    continue
                seq_lines.append(line.strip())

    return "".join(seq_lines).upper()


def _write_fasta(seq, path, header="genome"):
    """Write a single sequence to a FASTA file, wrapped at 70 columns."""
    with open(path, "w") as fh:
        fh.write(f">{header}\n")
        for i in range(0, len(seq), 70):
            fh.write(seq[i:i + 70] + "\n")


def _require_mummer():
    """
    Check that the MUMmer toolkit (nucmer + dnadiff) is available on
    PATH. Nucleotide diversity from unaligned whole genomes requires a
    genuine pairwise whole-genome alignment; MUMmer is the standard,
    scalable tool for this in bacterial population genomics.
    Install via: conda install -c bioconda mummer
    """
    for tool in ("nucmer", "dnadiff"):
        if shutil.which(tool) is None:
            raise RuntimeError(
                f"'{tool}' not found on PATH. Nucleotide diversity "
                "calculation requires MUMmer (nucmer/dnadiff) for "
                "whole-genome pairwise alignment. Install it, e.g. "
                "'conda install -c bioconda mummer', and ensure it is "
                "on PATH."
            )


def pairwise_genome_pi(fasta_a, fasta_b, workdir):
    """
    Align two whole genomes with MUMmer's dnadiff (nucmer + delta-filter
    + SNP calling under the hood) and return the pairwise nucleotide
    diversity contribution: SNPs / aligned_bases.

    dnadiff reports, among other things:
      - AlignedBases: total reference bases covered by 1-to-1 alignments
      - TotalSNPs:    total SNPs within those 1-to-1 aligned regions

    Restricting to 1-to-1 aligned blocks avoids counting differences
    across repeats/rearrangements as if they were homologous sites,
    which a naive Hamming distance on unaligned/concatenated sequence
    would not guard against.
    """
    prefix = os.path.join(workdir, "dnadiff_out")

    result = subprocess.run(
        ["dnadiff", "-p", prefix, fasta_a, fasta_b],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    report_path = prefix + ".report"
    if result.returncode != 0 or not os.path.exists(report_path):
        return np.nan, 0

    total_snps = None
    aligned_bases = None

    with open(report_path) as fh:
        for line in fh:
            if line.startswith("TotalSNPs"):
                parts = line.split()
                total_snps = float(parts[1])
            elif line.startswith("AlignedBases"):
                # format: "AlignedBases   123456(98.76%)  123456(97.65%)"
                parts = line.split()
                match = re.match(r"(\d+)", parts[1])
                if match:
                    aligned_bases = float(match.group(1))

    if total_snps is None or not aligned_bases:
        return np.nan, 0

    return total_snps / aligned_bases, aligned_bases


def seed_nucleotide_diversity(seed_gff_files):
    """
    Given the GFF files for all isolates of one seed, extract each
    isolate's genome sequence, align every pair of genomes with MUMmer
    (dnadiff), and compute nucleotide diversity (pi) as the mean, over
    all pairwise comparisons, of (SNPs / alignable sites). This is the
    standard pairwise-difference estimator of pi for a set of unaligned
    whole genomes.
    """
    _require_mummer()

    sequences = []
    for gff_file in seed_gff_files:
        seq = read_gff_sequence(gff_file)
        if seq:
            sequences.append((gff_file, seq))

    if len(sequences) < 2:
        return np.nan

    with tempfile.TemporaryDirectory() as tmp_root:

        fasta_paths = []
        for i, (gff_file, seq) in enumerate(sequences):
            fasta_path = os.path.join(tmp_root, f"isolate_{i}.fasta")
            _write_fasta(seq, fasta_path, header=os.path.basename(gff_file))
            fasta_paths.append(fasta_path)

        pair_pis = []
        for i, j in combinations(range(len(fasta_paths)), 2):
            with tempfile.TemporaryDirectory(dir=tmp_root) as pair_dir:
                pi_ij, aligned_bases = pairwise_genome_pi(
                    fasta_paths[i], fasta_paths[j], pair_dir
                )
                if not np.isnan(pi_ij) and aligned_bases > 0:
                    pair_pis.append(pi_ij)

    return float(np.mean(pair_pis)) if pair_pis else np.nan


def compute_diversity_across_seeds(diversity_root, seeds):
    """
    For every known seed, find its GFF files under diversity_root,
    align all pairs of isolate genomes with MUMmer, and compute true
    nucleotide diversity (pi) as the mean pairwise SNPs-per-aligned-site.
    Returns a DataFrame with columns: seed, n_isolates, nucleotide_diversity.
    """
    rows = []

    all_gffs = find_gffs(diversity_root)
    all_gffs = [g for g in all_gffs if "iso" in os.path.basename(g).lower()]

    for seed in seeds:
        seed_gffs = [g for g in all_gffs if extract_seed(g) == seed]

        if not seed_gffs:
            continue

        print(f"Seed {seed}: {len(seed_gffs)} GFFs -> pairwise whole-genome alignment")

        div = seed_nucleotide_diversity(seed_gffs)

        rows.append({
            "seed": seed,
            "n_isolates": len(seed_gffs),
            "nucleotide_diversity": div
        })

    return pd.DataFrame(rows)


def plot_nucleotide_diversity(diversity_df, out):

    if diversity_df.empty:
        print("No diversity data to plot.")
        return

    diversity_df = diversity_df.sort_values("nucleotide_diversity")

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=diversity_df,
        x="seed",
        y="nucleotide_diversity",
        order=diversity_df["seed"].astype(str),
        color="steelblue"
    )

    plt.xticks(rotation=90, fontsize=7)
    plt.xlabel("Seed")
    plt.ylabel("Nucleotide diversity (\u03c0)\nmean pairwise SNPs / aligned site")
    plt.title("Nucleotide diversity across simulated populations")

    plt.tight_layout()
    plt.savefig(
        os.path.join(out, "nucleotide_diversity_by_seed.png"),
        dpi=300
    )
    plt.close()


# ---------------------------------------------------------
# BASIC METRICS
# ---------------------------------------------------------

def plot_gene_length_distribution(gff_df, out):

    lengths = (
        gff_df["end"] -
        gff_df["start"] + 1
    )

    plt.figure(figsize=(7,6))

    sns.histplot(
        lengths,
        bins=60,
        kde=True
    )

    plt.xlabel("Gene length (bp)")
    plt.ylabel("Count")

    plt.title(
        "Gene length distribution"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out,
            "gene_length_distribution.png"
        ),
        dpi=300
    )

    plt.close()

def plot_gene_density(gff_df, out):

    midpoint = (
        gff_df["start"] +
        gff_df["end"]
    ) / 2

    plt.figure(figsize=(10,4))

    sns.histplot(
        midpoint,
        bins=100
    )

    plt.xlabel(
        "Genome position"
    )

    plt.ylabel(
        "Number of genes"
    )

    plt.title(
        "Gene density across genome"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out,
            "gene_density.png"
        ),
        dpi=300
    )

    plt.close()

def plot_strand_bias(gff_df, out):

    counts = gff_df["strand"].value_counts()

    plt.figure(figsize=(5,5))

    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%"
    )

    plt.title(
        "Strand orientation"
    )

    plt.savefig(
        os.path.join(
            out,
            "strand_bias.png"
        ),
        dpi=300
    )

    plt.close()

def plot_intergenic_distances(
    gff_df,
    out
):

    gff_df = gff_df.sort_values(
        "start"
    )

    distances = []

    ends = gff_df["end"].values
    starts = gff_df["start"].values

    for i in range(
        len(gff_df)-1
    ):

        distances.append(
            max(
                0,
                starts[i+1] - ends[i]
            )
        )

    plt.figure(
        figsize=(7,5)
    )

    sns.histplot(
        distances,
        bins=60
    )

    plt.xlabel(
        "Intergenic distance (bp)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Intergenic spacing"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out,
            "intergenic_distances.png"
        ),
        dpi=300
    )

    plt.close()

def plot_genome_architecture(
    gff_df,
    out,
    max_genes=300
):

    gff_df = gff_df.sort_values(
        "start"
    )

    subset = gff_df.iloc[:max_genes]

    plt.figure(
        figsize=(14,3)
    )

    for _, row in subset.iterrows():

        color = (
            "royalblue"
            if row["strand"] == "+"
            else "firebrick"
        )

        plt.plot(
            [row["start"], row["end"]],
            [0, 0],
            lw=5,
            color=color
        )

    plt.yticks([])

    plt.xlabel(
        "Genome position (bp)"
    )

    plt.title(
        "Genome architecture"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out,
            "genome_architecture.png"
        ),
        dpi=300
    )

    plt.close()

def accumulation(mat):

    out = []

    for i in range(1, mat.shape[1] + 1):

        sub = mat.iloc[:, :i]

        out.append(
            (sub.sum(axis=1) > 0).sum()
        )

    return out


def core_decay(mat):

    out = []

    for i in range(1, mat.shape[1] + 1):

        sub = mat.iloc[:, :i]

        out.append(
            (sub.sum(axis=1) == i).sum()
        )

    return out


def jaccard_distribution(mat):

    vals = []

    sets = [
        set(mat.index[mat[col] == 1])
        for col in mat.columns
    ]

    for a, b in combinations(sets, 2):

        vals.append(
            len(a & b) / len(a | b)
        )

    return vals


# ---------------------------------------------------------
# FIGURE 1
# RAREFACTION CLOUD
# ---------------------------------------------------------

def plot_rarefaction_cloud(accumulations, out, seeds=None):

    arr = np.array(accumulations)

    mean = arr.mean(axis=0)
    sd = arr.std(axis=0)

    x = np.arange(1, len(mean) + 1)

    plt.figure(figsize=(8, 6))

    if seeds is not None:
        palette = sns.color_palette("husl", len(seeds))
        for curve, seed, colour in zip(arr, seeds, palette):
            plt.plot(x, curve, alpha=0.35, color=colour, label=str(seed))
    else:
        for curve in arr:
            plt.plot(x, curve, alpha=0.20)

    plt.plot(
        x,
        mean,
        color="black",
        linewidth=3,
        label="Mean"
    )

    plt.fill_between(
        x,
        mean - sd,
        mean + sd,
        alpha=0.3
    )

    if seeds is not None:
        plt.legend(
            title="Seed",
            fontsize=6,
            ncol=2,
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

    plt.xlabel("Number of isolates")
    plt.ylabel("Pangenome size")
    plt.title("Pangenome rarefaction (per seed)")

    plt.tight_layout()
    plt.savefig(
        os.path.join(out, "rarefaction_cloud.png"),
        dpi=300
    )
    plt.close()


# ---------------------------------------------------------
# FIGURE 2
# CORE DECAY
# ---------------------------------------------------------

def plot_core_decay(core_curves, out, seeds=None):

    arr = np.array(core_curves)

    mean = arr.mean(axis=0)
    sd = arr.std(axis=0)

    x = np.arange(1, len(mean) + 1)

    plt.figure(figsize=(8, 6))

    if seeds is not None:
        palette = sns.color_palette("husl", len(seeds))
        for curve, seed, colour in zip(arr, seeds, palette):
            plt.plot(x, curve, alpha=0.35, color=colour, label=str(seed))
    else:
        for curve in arr:
            plt.plot(x, curve, alpha=0.2)

    plt.plot(
        x,
        mean,
        color="black",
        lw=3,
        label="Mean"
    )

    plt.fill_between(
        x,
        mean - sd,
        mean + sd,
        alpha=0.3
    )

    if seeds is not None:
        plt.legend(
            title="Seed",
            fontsize=6,
            ncol=2,
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

    plt.xlabel("Number of isolates")
    plt.ylabel("Core genes")
    plt.title("Core genome decay (per seed)")

    plt.tight_layout()
    plt.savefig(
        os.path.join(out, "core_decay.png"),
        dpi=300
    )
    plt.close()


# ---------------------------------------------------------
# FIGURE 3
# FREQUENCY SPECTRUM
# ---------------------------------------------------------

def plot_frequency_spectrum(mat, out):

    freq = mat.sum(axis=1)

    plt.figure(figsize=(8, 6))

    plt.hist(
        freq,
        bins=np.arange(1, mat.shape[1] + 2),
        log=True
    )

    plt.xlabel("Number of isolates carrying gene")
    plt.ylabel("Genes (log scale)")
    plt.title("Gene frequency spectrum")

    plt.tight_layout()
    plt.savefig(
        os.path.join(out, "frequency_spectrum.png"),
        dpi=300
    )
    plt.close()


# ---------------------------------------------------------
# FIGURE 4
# RANK ABUNDANCE
# ---------------------------------------------------------

def plot_rank_abundance(mat, out):

    freq = mat.sum(axis=1)

    freq = np.sort(freq.values)[::-1]

    plt.figure(figsize=(8, 6))

    plt.plot(
        np.arange(1, len(freq) + 1),
        freq
    )

    plt.yscale("log")

    plt.xlabel("Gene rank")
    plt.ylabel("Gene prevalence")
    plt.title("Rank-abundance curve")

    plt.tight_layout()

    plt.savefig(
        os.path.join(out, "rank_abundance.png"),
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# FIGURE 5
# CORE SHELL CLOUD
# ---------------------------------------------------------

def plot_core_shell_cloud(mat, out):

    n = mat.shape[1]

    prevalence = mat.sum(axis=1) / n

    core = (prevalence >= 0.95).sum()

    shell = (
        (prevalence >= 0.15) &
        (prevalence < 0.95)
    ).sum()

    cloud = (
        prevalence < 0.15
    ).sum()

    plt.figure(figsize=(6, 6))

    plt.pie(
        [core, shell, cloud],
        labels=[
            f"Core ({core})",
            f"Shell ({shell})",
            f"Cloud ({cloud})"
        ],
        autopct="%1.1f%%"
    )

    plt.title("Pangenome composition")

    plt.savefig(
        os.path.join(out, "core_shell_cloud.png"),
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# FIGURE 6
# PCA
# ---------------------------------------------------------

def plot_pca(mat, out):

    X = mat.T.values

    pca = PCA(n_components=2)

    scores = pca.fit_transform(X)

    pc1 = pca.explained_variance_ratio_[0] * 100
    pc2 = pca.explained_variance_ratio_[1] * 100

    plt.figure(figsize=(7,6))

    plt.scatter(
        scores[:,0],
        scores[:,1],
        alpha=0.7,
        s=30
    )

    plt.xlabel(f"PC1 ({pc1:.1f}%)")
    plt.ylabel(f"PC2 ({pc2:.1f}%)")

    plt.title("PCA of isolates")

    plt.tight_layout()

    plt.savefig(
        os.path.join(out,"pca_isolates.png"),
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# FIGURE 7
# JACCARD HEATMAP
# ---------------------------------------------------------

def plot_jaccard_heatmap(mat, out):

    X = mat.T.values

    dist = pdist(
        X,
        metric="jaccard"
    )

    similarity = 1 - squareform(dist)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        similarity,
        cmap="viridis"
    )

    plt.title("Jaccard similarity heatmap")

    plt.tight_layout()

    plt.savefig(
        os.path.join(out, "jaccard_heatmap.png"),
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# FIGURE 8
# DENDROGRAM
# ---------------------------------------------------------

def plot_dendrogram(mat, out):

    X = mat.T.values

    Z = linkage(
        X,
        method="average",
        metric="jaccard"
    )

    plt.figure(figsize=(12, 5))

    dendrogram(
        Z,
        no_labels=True
    )

    plt.title("Hierarchical clustering")

    plt.tight_layout()

    plt.savefig(
        os.path.join(out, "genome_dendrogram.png"),
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# FIGURE 9
# HEAPS LAW
# ---------------------------------------------------------

def heaps(x, k, alpha):
    return k * (x ** alpha)


def estimate_heaps(accumulations):

    alphas = []

    for curve in accumulations:

        x = np.arange(
            1,
            len(curve) + 1
        )

        try:

            popt, _ = curve_fit(
                heaps,
                x,
                curve,
                maxfev=10000
            )

            alphas.append(
                popt[1]
            )

        except Exception:
            pass

    return alphas


def plot_openness(accumulations, out, seeds=None):

    alphas = estimate_heaps(accumulations)

    plt.figure(figsize=(6, 6))

    sns.violinplot(
        y=alphas,
        inner=None,
        color="steelblue"
    )

    sns.stripplot(
        y=alphas,
        color="black",
        size=5,
        jitter=0.05
    )

    plt.ylabel("Heaps alpha")
    plt.title("Pangenome openness (one point per seed)")

    plt.tight_layout()

    plt.savefig(
        os.path.join(out, "heaps_alpha.png"),
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# FIGURE 10
# PHASE SPACE
# ---------------------------------------------------------

def core_accessory_trajectory(mat):

    core = []
    accessory = []

    for i in range(1, mat.shape[1] + 1):

        sub = mat.iloc[:, :i]

        freq = sub.sum(axis=1)

        core.append(
            (freq == i).sum()
        )

        accessory.append(
            ((freq > 0) & (freq < i)).sum()
        )

    return core, accessory


def plot_phase_space(mats, out):

    plt.figure(figsize=(7, 6))

    for mat in mats:

        core, accessory = \
            core_accessory_trajectory(mat)

        plt.plot(
            core,
            accessory,
            alpha=0.3
        )

    plt.xlabel("Core genes")
    plt.ylabel("Accessory genes")
    plt.title("Pangenome phase space")

    plt.tight_layout()

    plt.savefig(
        os.path.join(out, "phase_space.png"),
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def analyse(
    folder,
    gff_folder=None,
    diversity_root=None
):

    out = os.path.join(
        folder,
        "figures"
    )

    os.makedirs(out, exist_ok=True)

    files = find_files(folder)

    mats = []
    accumulations = []
    cores = []
    pangenome_sizes = []
    seeds = []

    for f in files:

        print("Loading:", f)

        seed = extract_seed(f)
        if seed is None:
            print(f"  WARNING: could not identify seed for {f}, skipping")
            continue

        mat = load_pa(f)

        mats.append(mat)
        seeds.append(seed)

        accumulations.append(
            accumulation(mat)
        )

        cores.append(
            core_decay(mat)
        )

        pangenome_sizes.append(
            len(mat)
        )

    if not mats:
        raise RuntimeError("No presence/absence files with a recognised seed were found.")

    representative = mats[0]

    plot_rarefaction_cloud(
        accumulations,
        out,
        seeds=seeds
    )

    plot_core_decay(
        cores,
        out,
        seeds=seeds
    )

    plot_frequency_spectrum(
        representative,
        out
    )

    plot_rank_abundance(
        representative,
        out
    )

    plot_core_shell_cloud(
        representative,
        out
    )

    plot_pca(
        representative,
        out
    )

    plot_jaccard_heatmap(
        representative,
        out
    )

    plot_dendrogram(
        representative,
        out
    )

    plot_openness(
        accumulations,
        out,
        seeds=seeds
    )

    plot_phase_space(
        mats,
        out
    )

    pd.DataFrame({
        "seed": seeds,
        "pangenome_size": pangenome_sizes
    }).to_csv(
        os.path.join(
            folder,
            "summary_statistics.csv"
        ),
        index=False
    )

    # ---------------------------------------------------------
    # Nucleotide diversity across the 30 populations, computed from
    # the GFF-embedded genome sequences of each seed's 100 isolates.
    # ---------------------------------------------------------
    if diversity_root is not None:

        diversity_df = compute_diversity_across_seeds(
            diversity_root,
            SEEDS
        )

        diversity_df.to_csv(
            os.path.join(folder, "nucleotide_diversity_by_seed.csv"),
            index=False
        )

        plot_nucleotide_diversity(diversity_df, out)

    if gff_folder is not None:

        gffs = find_gffs(
            gff_folder
        )

        if len(gffs):

            print(
                "Loading GFF:",
                gffs[0]
            )

            gff_df = read_gff(
                gffs[0]
            )

            plot_gene_length_distribution(
                gff_df,
                out
            )

            plot_gene_density(
                gff_df,
                out
            )

            plot_strand_bias(
                gff_df,
                out
            )

            plot_genome_architecture(
                gff_df,
                out
            )

            plot_intergenic_distances(
                gff_df,
                out
            )

    print("Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        required=False,
        default = "/nfs/research/jlees/campan/data/clustering_benchmarking/2026_06_10_simsnowwithntandaasandgffs/simulations/PROKKA_06122025",
    )

    parser.add_argument(
        "-g",
        "--gff-folder",
        required=False,
        default="/nfs/research/jlees/campan/data/clustering_benchmarking/2026_06_10_simsnowwithntandaasandgffs/MSdataset/6925_1#61/",
        help="Folder containing original (reference) GFF files, used for the gene-architecture plots"
    )

    parser.add_argument(
        "-d",
        "--diversity-root",
        required=False,
        default="/nfs/research/jlees/campan/data/clustering_benchmarking/2026_06_10_simsnowwithntandaasandgffs/simulations/PROKKA_06122025",
        help=(
            "Root folder containing the 30 per-seed subfolders, each with "
            "~100 simulated isolate GFFs (with embedded ##FASTA), used to "
            "compute true nucleotide diversity (pi) per seed via pairwise "
            "MUMmer whole-genome alignment. Set to '' to skip."
        )
    )

    args = parser.parse_args()

    analyse(
        args.input,
        gff_folder=args.gff_folder,
        diversity_root=(args.diversity_root or None)
    )
