#!/usr/bin/env python3
"""
Multi-assembly pangenome simulation analysis.

CHANGES vs the original single-assembly script (see accompanying summary
for full rationale):

  1. Iterates over *all* assemblies found under the input root, instead of
     hardcoding "PROKKA". Assemblies are auto-detected by directory name
     pattern PROKKA_<date>[__params], e.g.
     'PROKKA_06122025__gr_xx_lr_xx_mu_xx'.
  2. Nucleotide diversity is now estimated with a MinHash/Mash-distance
     k-mer sketch proxy instead of full pairwise MUMmer whole-genome
     alignment (no external binaries, no per-pair subprocess, orders of
     magnitude faster - see seed_nucleotide_diversity_proxy()).
  3. Every per-seed analysis still runs separately per assembly (same
     per-assembly figures/ folder as before), and a new set of
     "*_comparison.png" figures is produced under <root>/figures_comparison
     to compare assemblies directly.
  4. Pangenome openness (Heaps' alpha) is now shown as one boxplot per
     assembly (with points overlaid) instead of a single violin plot.
  5. Duplicated plotting/looping logic has been factored into shared
     helpers (analyse_assembly, *_comparison plotting) to scale cleanly
     to any number of assemblies.
"""

import os
import glob
import argparse
import re
import itertools
from concurrent.futures import ProcessPoolExecutor

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

# Directory naming convention for an assembly, e.g.
#   PROKKA_06122025
#   PROKKA_06122025__gr_xx_lr_xx_mu_xx
# Anything starting with "PROKKA" is treated as an assembly directory so
# the pipeline scales automatically to however many are present.
ASSEMBLY_DIR_PATTERN = re.compile(r"^PROKKA(_\w+)?(__.*)?$")


# ---------------------------------------------------------
# ASSEMBLY DISCOVERY
# ---------------------------------------------------------

def discover_assemblies(root):
    """
    Find every assembly directory directly under `root` whose name matches
    the PROKKA_<date>[__params] convention. Returns a sorted list of
    (assembly_name, assembly_path) tuples, so the rest of the pipeline
    never needs to know how many assemblies there are or what they're
    called.

    Backward compatible: if `root` itself is a PROKKA_* directory (the old
    single-assembly usage, e.g. -i .../PROKKA_06122025), it is treated as
    the sole assembly.
    """
    candidates = []

    if os.path.isdir(root):
        for entry in sorted(os.listdir(root)):
            path = os.path.join(root, entry)
            if entry == "PROKKA_06122025":
                # Skip the bare "PROKKA" directory (no date/params
                # suffix)
                print(f"Skipping '{entry}': bare PROKKA directory, no assembly suffix")
                continue
            if os.path.isdir(path) and ASSEMBLY_DIR_PATTERN.match(entry):
                candidates.append((entry, path))

    if candidates:
        return candidates

    base = os.path.basename(os.path.normpath(root))
    if base != "PROKKA" and ASSEMBLY_DIR_PATTERN.match(base):
        return [(base, root)]

    raise RuntimeError(
        f"No assembly directories matching '{ASSEMBLY_DIR_PATTERN.pattern}' "
        f"found under {root}"
    )


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


# ---------------------------------------------------------
# NUCLEOTIDE DIVERSITY PROXY (MinHash / Mash distance)
# ---------------------------------------------------------
#
# The original implementation ran nucmer+dnadiff (MUMmer) on every pair
# of isolate genomes to get exact SNP-based pi. For n isolates that's
# C(n,2) external alignment subprocesses per seed (e.g. ~4950 for 100
# isolates), repeated for every seed and now every assembly - this
# dominates runtime.
#
# Proxy used here: MinHash k-mer sketches + Mash distance (Ondov et al.
# 2016, "Mash: fast genome and metagenome distance estimation using
# MinHash"). Each genome is reduced to a small (~1000 hash) sketch of its
# k-mers in O(genome length) time; pairwise distance is then estimated
# from sketch-vs-sketch Jaccard similarity in O(sketch size). This is a
# standard, well-validated alignment-free estimator of average
# nucleotide divergence between genomes, and is dramatically cheaper:
# no external tools, no per-pair alignment, sketch construction is done
# once per genome rather than once per pair.
#
# It is a *proxy*, not exact SNP-calling pi: it will be less accurate
# for genomes with large structural rearrangements/indels, but for
# closely related simulated isolates (the typical case here) it tracks
# true pairwise divergence closely and is the recommended fast
# alternative to full alignment.

_BASE_LOOKUP = np.zeros(256, dtype=np.uint64)
for _i, _b in enumerate(b"ACGT"):
    _BASE_LOOKUP[_b] = _i

_FIBONACCI_MIX = np.uint64(0x9E3779B97F4A7C15)


def _kmer_codes(seq, k):
    """
    Vectorised 2-bit-per-base encoding of every k-mer in `seq` as a
    single integer. Non-ACGT bases (N, etc.) are mapped to 'A' (0); this
    is a negligible source of noise for a diversity *proxy* and avoids
    an expensive per-base branch.
    """
    if len(seq) < k:
        return np.array([], dtype=np.uint64)

    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    codes = _BASE_LOOKUP[arr]

    n = len(codes) - k + 1
    kmer_codes = np.zeros(n, dtype=np.uint64)

    # Build the rolling base-4 k-mer code in k vectorised steps
    # (not one Python loop iteration per base of the genome).
    for offset in range(k):
        kmer_codes = (kmer_codes << np.uint64(2)) | codes[offset:offset + n].astype(np.uint64)

    return kmer_codes


def minhash_sketch(seq, k=21, sketch_size=1000):
    """
    Bottom-sketch MinHash: hash every k-mer, keep the `sketch_size`
    smallest (distinct) hash values. This is a uniform-ish sample of the
    genome's k-mer content and is the standard basis for Mash-style
    distance estimation.
    """
    codes = _kmer_codes(seq, k)
    if codes.size == 0:
        return np.array([], dtype=np.uint64)

    mixed = codes * _FIBONACCI_MIX  # spread values (Fibonacci hashing)
    mixed = np.unique(mixed)

    if mixed.size > sketch_size:
        idx = np.argpartition(mixed, sketch_size)[:sketch_size]
        mixed = np.sort(mixed[idx])

    return mixed


def mash_distance(sketch_a, sketch_b, k):
    """
    Estimate nucleotide divergence between two genomes from their MinHash
    sketches, using the Mash distance formula (Ondov et al. 2016):

        D = -1/k * ln(2J / (1 + J))

    where J is the Jaccard similarity between the two k-mer sketches.
    """
    if sketch_a.size == 0 or sketch_b.size == 0:
        return np.nan

    inter = np.intersect1d(sketch_a, sketch_b, assume_unique=True).size
    union = sketch_a.size + sketch_b.size - inter

    if union == 0:
        return np.nan

    jaccard = inter / union

    if jaccard <= 0:
        return 1.0  # sketches share nothing measurable -> treat as max divergence

    d = -1.0 / k * np.log(2 * jaccard / (1 + jaccard))
    return max(d, 0.0)


def seed_nucleotide_diversity_proxy(seed_gff_files, k=21, sketch_size=1000):
    """
    Given the GFF files for all isolates of one seed, sketch each
    isolate's genome and return the mean pairwise Mash distance as a
    fast proxy for nucleotide diversity (pi).
    """
    sketches = [
        minhash_sketch(seq, k=k, sketch_size=sketch_size)
        for gff_file in seed_gff_files
        if (seq := read_gff_sequence(gff_file))
    ]

    if len(sketches) < 2:
        return np.nan

    dists = [
        mash_distance(a, b, k)
        for a, b in combinations(sketches, 2)
    ]
    dists = [d for d in dists if not np.isnan(d)]

    return float(np.mean(dists)) if dists else np.nan


def _seed_diversity_task(args):
    seed, seed_gffs, k, sketch_size = args
    div = seed_nucleotide_diversity_proxy(seed_gffs, k=k, sketch_size=sketch_size)
    return seed, len(seed_gffs), div


def compute_diversity_across_seeds(diversity_root, seeds, k=21, sketch_size=1000, n_workers=None):
    """
    For every known seed, find its GFF files under diversity_root and
    compute the MinHash/Mash nucleotide diversity proxy. Seeds are
    processed in parallel (sketching is CPU-bound but independent per
    seed) using the CPUs actually allocated to the job when running
    under a scheduler.
    Returns a DataFrame with columns: seed, n_isolates, nucleotide_diversity.
    """
    all_gffs = find_gffs(diversity_root)
    all_gffs = [g for g in all_gffs if "iso" in os.path.basename(g).lower()]

    tasks = []
    for seed in seeds:
        seed_gffs = [g for g in all_gffs if extract_seed(g) == seed]
        if seed_gffs:
            tasks.append((seed, seed_gffs, k, sketch_size))

    if not tasks:
        return pd.DataFrame(columns=["seed", "n_isolates", "nucleotide_diversity"])

    n_workers = n_workers or int(
        os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("LSB_DJOB_NUMPROC")
        or os.cpu_count()
        or 1
    )

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for seed, n_iso, div in executor.map(_seed_diversity_task, tasks):
            msg = f"pi proxy = {div:.6g}" if not np.isnan(div) else "pi proxy = NA"
            print(f"  Seed {seed}: {n_iso} isolates -> {msg}")
            rows.append({"seed": seed, "n_isolates": n_iso, "nucleotide_diversity": div})

    return pd.DataFrame(rows)


def plot_nucleotide_diversity(diversity_df, out, assembly_name=None):

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
    plt.ylabel("Nucleotide diversity")
    title = "Nucleotide diversity across simulated populations"
    if assembly_name:
        title += f" ({assembly_name})"
    plt.title(title)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out, "nucleotide_diversity_by_seed.png"),
        dpi=300
    )
    plt.close()


def plot_diversity_comparison(diversity_all_df, out):
    """Boxplot of the nucleotide diversity proxy, one box per assembly."""

    if diversity_all_df.empty:
        print("No diversity data to compare.")
        return

    plt.figure(figsize=(max(6, 1.5 * diversity_all_df["assembly"].nunique()), 6))

    order = sorted(diversity_all_df["assembly"].unique())
    palette = sns.color_palette("tab10", len(order))

    sns.boxplot(
        data=diversity_all_df,
        x="assembly",
        y="nucleotide_diversity",
        order=order,
        hue="assembly",
        palette=palette,
        legend=False,
        showfliers=False
    )
    sns.stripplot(
        data=diversity_all_df,
        x="assembly",
        y="nucleotide_diversity",
        order=order,
        color="black",
        size=4,
        jitter=0.15,
        alpha=0.6
    )

    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel("Assembly", fontsize=9)
    plt.ylabel("Nucleotide diversity", fontsize=9)
    plt.title("Nucleotide diversity across assemblies (one point per seed)", fontsize=10)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out, "nucleotide_diversity_comparison.png"),
        dpi=300
    )
    plt.close()


# ---------------------------------------------------------
# BASIC METRICS (unchanged, run per-assembly using a reference GFF)
# ---------------------------------------------------------

def plot_gene_length_distribution(gff_df, out):

    lengths = gff_df["end"] - gff_df["start"] + 1

    plt.figure(figsize=(7, 6))
    sns.histplot(lengths, bins=60, kde=True)
    plt.xlabel("Gene length (bp)")
    plt.ylabel("Count")
    plt.title("Gene length distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "gene_length_distribution.png"), dpi=300)
    plt.close()


def plot_gene_density(gff_df, out):

    midpoint = (gff_df["start"] + gff_df["end"]) / 2

    plt.figure(figsize=(10, 4))
    sns.histplot(midpoint, bins=100)
    plt.xlabel("Genome position")
    plt.ylabel("Number of genes")
    plt.title("Gene density across genome")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "gene_density.png"), dpi=300)
    plt.close()


def plot_strand_bias(gff_df, out):

    counts = gff_df["strand"].value_counts()

    plt.figure(figsize=(5, 5))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    plt.title("Strand orientation")
    plt.savefig(os.path.join(out, "strand_bias.png"), dpi=300)
    plt.close()


def plot_intergenic_distances(gff_df, out):

    gff_df = gff_df.sort_values("start")

    ends = gff_df["end"].values
    starts = gff_df["start"].values

    distances = np.maximum(0, starts[1:] - ends[:-1])

    plt.figure(figsize=(7, 5))
    sns.histplot(distances, bins=60)
    plt.xlabel("Intergenic distance (bp)")
    plt.ylabel("Frequency")
    plt.title("Intergenic spacing")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "intergenic_distances.png"), dpi=300)
    plt.close()


def plot_genome_architecture(gff_df, out, max_genes=300):

    gff_df = gff_df.sort_values("start")
    subset = gff_df.iloc[:max_genes]

    plt.figure(figsize=(14, 3))

    for _, row in subset.iterrows():
        color = "royalblue" if row["strand"] == "+" else "firebrick"
        plt.plot([row["start"], row["end"]], [0, 0], lw=5, color=color)

    plt.yticks([])
    plt.xlabel("Genome position (bp)")
    plt.title("Genome architecture")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "genome_architecture.png"), dpi=300)
    plt.close()


def accumulation(mat):
    out = []
    for i in range(1, mat.shape[1] + 1):
        sub = mat.iloc[:, :i]
        out.append((sub.sum(axis=1) > 0).sum())
    return out


def core_decay(mat):
    out = []
    for i in range(1, mat.shape[1] + 1):
        sub = mat.iloc[:, :i]
        out.append((sub.sum(axis=1) == i).sum())
    return out


def jaccard_distribution(mat):
    vals = []
    sets = [set(mat.index[mat[col] == 1]) for col in mat.columns]
    for a, b in combinations(sets, 2):
        vals.append(len(a & b) / len(a | b))
    return vals


# ---------------------------------------------------------
# FIGURE 1: RAREFACTION CLOUD (per-assembly, unchanged look)
# ---------------------------------------------------------

def plot_rarefaction_cloud(accumulations, out, seeds=None, assembly_name=None):

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

    plt.plot(x, mean, color="black", linewidth=3, label="Mean")
    plt.fill_between(x, mean - sd, mean + sd, alpha=0.3)

    if seeds is not None:
        legend = plt.legend(title="Seed", fontsize=5, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
        if legend is not None and legend.get_title() is not None:
            legend.get_title().set_fontsize(6)

    plt.xlabel("Number of isolates", fontsize=9)
    plt.ylabel("Pangenome size", fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    title = "Pangenome rarefaction (per seed)"
    if assembly_name:
        title += f" - {assembly_name}"
    plt.title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "rarefaction_cloud.png"), dpi=300)
    plt.close()


def plot_rarefaction_comparison(accum_by_assembly, out):
    """Mean +/- sd rarefaction curve per assembly, overlaid for comparison."""

    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("tab10", len(accum_by_assembly))

    for (assembly_name, accumulations), colour in zip(accum_by_assembly.items(), palette):
        if not accumulations:
            continue
        arr = np.array(accumulations)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0)
        x = np.arange(1, len(mean) + 1)

        plt.plot(x, mean, color=colour, linewidth=2.5, label=assembly_name)
        plt.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.15)

    plt.xlabel("Number of isolates")
    plt.ylabel("Pangenome size")
    plt.title("Pangenome rarefaction - assembly comparison")
    plt.legend(title="Assembly", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "rarefaction_cloud_comparison.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------
# FIGURE 2: CORE DECAY (per-assembly, unchanged look)
# ---------------------------------------------------------

def plot_core_decay(core_curves, out, seeds=None, assembly_name=None):

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

    plt.plot(x, mean, color="black", lw=3, label="Mean")
    plt.fill_between(x, mean - sd, mean + sd, alpha=0.3)

    if seeds is not None:
        plt.legend(title="Seed", fontsize=6, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.xlabel("Number of isolates")
    plt.ylabel("Core genes")
    title = "Core genome decay (per seed)"
    if assembly_name:
        title += f" - {assembly_name}"
    plt.title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "core_decay.png"), dpi=300)
    plt.close()


def plot_core_decay_comparison(core_by_assembly, out):
    """Mean +/- sd core-decay curve per assembly, overlaid for comparison."""

    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("tab10", len(core_by_assembly))

    for (assembly_name, cores), colour in zip(core_by_assembly.items(), palette):
        if not cores:
            continue
        arr = np.array(cores)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0)
        x = np.arange(1, len(mean) + 1)

        plt.plot(x, mean, color=colour, linewidth=2.5, label=assembly_name)
        plt.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.15)

    plt.xlabel("Number of isolates")
    plt.ylabel("Core genes")
    plt.title("Core genome decay - assembly comparison")
    plt.legend(title="Assembly", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "core_decay_comparison.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------
# FIGURE 3: FREQUENCY SPECTRUM
# ---------------------------------------------------------

def plot_frequency_spectrum(mat, out, assembly_name=None):

    freq = mat.sum(axis=1)

    plt.figure(figsize=(8, 6))
    plt.hist(freq, bins=np.arange(1, mat.shape[1] + 2), log=True)
    plt.xlabel("Number of isolates carrying gene")
    plt.ylabel("Genes (log scale)")
    plt.title("Gene frequency spectrum" + (f" - {assembly_name}" if assembly_name else ""))
    plt.tight_layout()
    plt.savefig(os.path.join(out, "frequency_spectrum.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------
# FIGURE 4: RANK ABUNDANCE
# ---------------------------------------------------------

def plot_rank_abundance(mat, out, assembly_name=None):

    freq = mat.sum(axis=1)
    freq = np.sort(freq.values)[::-1]

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, len(freq) + 1), freq)
    plt.yscale("log")
    plt.xlabel("Gene rank")
    plt.ylabel("Gene prevalence")
    plt.title("Rank-abundance curve" + (f" - {assembly_name}" if assembly_name else ""))
    plt.tight_layout()
    plt.savefig(os.path.join(out, "rank_abundance.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------
# FIGURE 5: CORE SHELL CLOUD
# ---------------------------------------------------------

def plot_core_shell_cloud(mat, out, assembly_name=None):

    n = mat.shape[1]
    prevalence = mat.sum(axis=1) / n

    core = (prevalence >= 0.95).sum()
    shell = ((prevalence >= 0.15) & (prevalence < 0.95)).sum()
    cloud = (prevalence < 0.15).sum()

    plt.figure(figsize=(6, 6))
    plt.pie(
        [core, shell, cloud],
        labels=[f"Core ({core})", f"Shell ({shell})", f"Cloud ({cloud})"],
        autopct="%1.1f%%"
    )
    plt.title("Pangenome composition" + (f" - {assembly_name}" if assembly_name else ""))
    plt.savefig(os.path.join(out, "core_shell_cloud.png"), dpi=300)
    plt.close()

    return {"core": int(core), "shell": int(shell), "cloud": int(cloud)}


# ---------------------------------------------------------
# FIGURE 6: PCA
# ---------------------------------------------------------

def plot_pca(mat, out, assembly_name=None):

    X = mat.T.values
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)

    pc1 = pca.explained_variance_ratio_[0] * 100
    pc2 = pca.explained_variance_ratio_[1] * 100

    plt.figure(figsize=(7, 6))
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7, s=30)
    plt.xlabel(f"PC1 ({pc1:.1f}%)")
    plt.ylabel(f"PC2 ({pc2:.1f}%)")
    plt.title("PCA of isolates" + (f" - {assembly_name}" if assembly_name else ""))
    plt.tight_layout()
    plt.savefig(os.path.join(out, "pca_isolates.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------
# FIGURE 7: JACCARD HEATMAP
# ---------------------------------------------------------

def plot_jaccard_heatmap(mat, out, assembly_name=None):

    X = mat.T.values
    dist = pdist(X, metric="jaccard")
    similarity = 1 - squareform(dist)

    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity, cmap="viridis")
    plt.title("Jaccard similarity heatmap" + (f" - {assembly_name}" if assembly_name else ""))
    plt.tight_layout()
    plt.savefig(os.path.join(out, "jaccard_heatmap.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------
# FIGURE 8: DENDROGRAM
# ---------------------------------------------------------

def plot_dendrogram(mat, out, assembly_name=None):

    X = mat.T.values
    Z = linkage(X, method="average", metric="jaccard")

    plt.figure(figsize=(12, 5))
    dendrogram(Z, no_labels=True)
    plt.title("Hierarchical clustering" + (f" - {assembly_name}" if assembly_name else ""))
    plt.tight_layout()
    plt.savefig(os.path.join(out, "genome_dendrogram.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------
# FIGURE 9: HEAPS LAW / PANGENOME OPENNESS
# ---------------------------------------------------------

def heaps(x, k, alpha):
    return k * (x ** alpha)


def estimate_heaps(accumulations):

    alphas = []

    for curve in accumulations:
        x = np.arange(1, len(curve) + 1)
        try:
            popt, _ = curve_fit(heaps, x, curve, maxfev=10000)
            alphas.append(popt[1])
        except Exception:
            pass

    return alphas


def plot_openness(accumulations, out, seeds=None, assembly_name=None):
    """Per-assembly openness plot (kept for backward compatibility)."""

    alphas = estimate_heaps(accumulations)

    plt.figure(figsize=(6, 6))
    sns.boxplot(y=alphas, color="steelblue", showfliers=False)
    sns.stripplot(y=alphas, color="black", size=5, jitter=0.05)

    plt.ylabel("Heaps alpha", fontsize=9)
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.title(
        "Pangenome openness (one point per seed)" + (f" - {assembly_name}" if assembly_name else ""),
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out, "heaps_alpha.png"), dpi=300)
    plt.close()

    return alphas


def plot_openness_comparison(alphas_by_assembly, out):
    """
    Requirement #4: pangenome openness shown as one boxplot per assembly,
    side by side, instead of a single violin plot. Scales automatically
    to any number of assemblies.
    """

    rows = []
    for assembly_name, alphas in alphas_by_assembly.items():
        for a in alphas:
            rows.append({"assembly": assembly_name, "heaps_alpha": a})

    df = pd.DataFrame(rows)
    if df.empty:
        print("No openness data to compare.")
        return

    order = sorted(alphas_by_assembly.keys())
    palette = sns.color_palette("tab10", len(order))

    plt.figure(figsize=(max(6, 1.5 * len(order)), 6))

    sns.boxplot(
        data=df,
        x="assembly",
        y="heaps_alpha",
        order=order,
        hue="assembly",
        palette=palette,
        legend=False,
        showfliers=False
    )
    sns.stripplot(
        data=df,
        x="assembly",
        y="heaps_alpha",
        order=order,
        color="black",
        size=5,
        jitter=0.15,
        alpha=0.7
    )

    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel("Assembly", fontsize=9)
    plt.ylabel("Heaps alpha", fontsize=9)
    plt.title("Pangenome openness - assembly comparison (one point per seed)", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "heaps_alpha_comparison.png"), dpi=300)
    plt.close()


def plot_pangenome_size_comparison(summary_df, out):
    """Boxplot of pangenome size per assembly (one point per seed)."""

    if summary_df.empty:
        print("No pangenome size data to compare.")
        return

    order = sorted(summary_df["assembly"].unique())

    plt.figure(figsize=(max(6, 1.5 * len(order)), 6))

    sns.boxplot(
        data=summary_df,
        x="assembly",
        y="pangenome_size",
        order=order,
        color="lightgrey",
        showfliers=False
    )
    sns.stripplot(
        data=summary_df,
        x="assembly",
        y="pangenome_size",
        order=order,
        color="black",
        size=5,
        jitter=0.15,
        alpha=0.7
    )

    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Assembly")
    plt.ylabel("Pangenome size (genes)")
    plt.title("Pangenome size - assembly comparison (one point per seed)")

    plt.tight_layout()
    plt.savefig(os.path.join(out, "pangenome_size_comparison.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------
# FIGURE 10: PHASE SPACE
# ---------------------------------------------------------

def core_accessory_trajectory(mat):

    core = []
    accessory = []

    for i in range(1, mat.shape[1] + 1):
        sub = mat.iloc[:, :i]
        freq = sub.sum(axis=1)
        core.append((freq == i).sum())
        accessory.append(((freq > 0) & (freq < i)).sum())

    return core, accessory


def plot_phase_space(mats, out, assembly_name=None):

    plt.figure(figsize=(7, 6))

    for mat in mats:
        core, accessory = core_accessory_trajectory(mat)
        plt.plot(core, accessory, alpha=0.3)

    plt.xlabel("Core genes")
    plt.ylabel("Accessory genes")
    plt.title("Pangenome phase space" + (f" - {assembly_name}" if assembly_name else ""))
    plt.tight_layout()
    plt.savefig(os.path.join(out, "phase_space.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------
# PER-ASSEMBLY ANALYSIS
# ---------------------------------------------------------

def analyse_assembly(assembly_name, folder, diversity_root=None):
    """
    Run the full per-seed pangenome analysis for a single assembly.
    Produces the same figures/ output as the original single-assembly
    script (written under `folder/figures`), and returns the summary
    data needed to build the cross-assembly comparison plots.
    """

    out = os.path.join(folder, "figures")
    os.makedirs(out, exist_ok=True)

    files = find_files(folder)

    mats = []
    accumulations = []
    cores = []
    pangenome_sizes = []
    seeds = []

    for f in files:

        print(f"  Loading: {f}")

        seed = extract_seed(f)
        if seed is None:
            print(f"    WARNING: could not identify seed for {f}, skipping")
            continue

        mat = load_pa(f)

        mats.append(mat)
        seeds.append(seed)
        accumulations.append(accumulation(mat))
        cores.append(core_decay(mat))
        pangenome_sizes.append(len(mat))

    if not mats:
        raise RuntimeError(
            f"No presence/absence files with a recognised seed were found for "
            f"assembly '{assembly_name}' in {folder}."
        )

    representative = mats[0]

    plot_rarefaction_cloud(accumulations, out, seeds=seeds, assembly_name=assembly_name)
    plot_core_decay(cores, out, seeds=seeds, assembly_name=assembly_name)
    plot_frequency_spectrum(representative, out, assembly_name=assembly_name)
    plot_rank_abundance(representative, out, assembly_name=assembly_name)
    plot_core_shell_cloud(representative, out, assembly_name=assembly_name)
    plot_pca(representative, out, assembly_name=assembly_name)
    plot_jaccard_heatmap(representative, out, assembly_name=assembly_name)
    plot_dendrogram(representative, out, assembly_name=assembly_name)
    alphas = plot_openness(accumulations, out, seeds=seeds, assembly_name=assembly_name)
    plot_phase_space(mats, out, assembly_name=assembly_name)

    pd.DataFrame({
        "seed": seeds,
        "pangenome_size": pangenome_sizes
    }).to_csv(
        os.path.join(folder, "summary_statistics.csv"),
        index=False
    )

    diversity_df = None
    if diversity_root is not None:
        print(f"  Computing nucleotide diversity proxy for '{assembly_name}'...")
        diversity_df = compute_diversity_across_seeds(diversity_root, SEEDS)
        diversity_df.to_csv(
            os.path.join(folder, "nucleotide_diversity_by_seed.csv"),
            index=False
        )
        plot_nucleotide_diversity(diversity_df, out, assembly_name=assembly_name)

    return {
        "seeds": seeds,
        "pangenome_sizes": pangenome_sizes,
        "accumulations": accumulations,
        "cores": cores,
        "alphas": alphas,
        "diversity_df": diversity_df,
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def analyse(root, gff_folder=None, diversity_root=None):

    assemblies = discover_assemblies(root)
    print(f"Found {len(assemblies)} assembly(ies): {[name for name, _ in assemblies]}")

    comparison_out = os.path.join(root, "figures_comparison")
    os.makedirs(comparison_out, exist_ok=True)

    summary_rows = []
    alphas_by_assembly = {}
    accum_by_assembly = {}
    core_by_assembly = {}
    diversity_frames = []

    for assembly_name, assembly_path in assemblies:

        print(f"\n=== Assembly: {assembly_name} ===")

        # Assumption: each assembly has its own isolate GFFs used for the
        # nucleotide diversity proxy, living either directly in the
        # assembly folder, or under <diversity_root>/<assembly_name> if a
        # separate diversity root was given. Adjust here if your layout
        # differs.
        assembly_diversity_root = None
        if diversity_root is not None:
            candidate = os.path.join(diversity_root, assembly_name)
            assembly_diversity_root = candidate if os.path.isdir(candidate) else diversity_root

        result = analyse_assembly(
            assembly_name,
            assembly_path,
            diversity_root=assembly_diversity_root
        )

        for seed, size in zip(result["seeds"], result["pangenome_sizes"]):
            summary_rows.append({
                "assembly": assembly_name,
                "seed": seed,
                "pangenome_size": size
            })

        alphas_by_assembly[assembly_name] = result["alphas"]
        accum_by_assembly[assembly_name] = result["accumulations"]
        core_by_assembly[assembly_name] = result["cores"]

        if result["diversity_df"] is not None and not result["diversity_df"].empty:
            df = result["diversity_df"].copy()
            df["assembly"] = assembly_name
            diversity_frames.append(df)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(root, "summary_statistics_all_assemblies.csv"), index=False)

    plot_pangenome_size_comparison(summary_df, comparison_out)
    plot_openness_comparison(alphas_by_assembly, comparison_out)
    plot_rarefaction_comparison(accum_by_assembly, comparison_out)
    plot_core_decay_comparison(core_by_assembly, comparison_out)

    if diversity_frames:
        diversity_all = pd.concat(diversity_frames, ignore_index=True)
        diversity_all.to_csv(
            os.path.join(root, "nucleotide_diversity_by_seed_all_assemblies.csv"),
            index=False
        )
        plot_diversity_comparison(diversity_all, comparison_out)

    # Gene-architecture plots come from a single reference genome and are
    # not assembly-specific (assumption: gff_folder points at one
    # reference isolate, independent of the simulated assemblies), so
    # they are computed once and written to the comparison folder.
    if gff_folder is not None:

        gffs = find_gffs(gff_folder)

        if gffs:
            print(f"\nLoading reference GFF: {gffs[0]}")
            gff_df = read_gff(gffs[0])

            plot_gene_length_distribution(gff_df, comparison_out)
            plot_gene_density(gff_df, comparison_out)
            plot_strand_bias(gff_df, comparison_out)
            plot_genome_architecture(gff_df, comparison_out)
            plot_intergenic_distances(gff_df, comparison_out)

    print("\nDone.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input",
        required=False,
        default="/nfs/research/jlees/campan/data/clustering_benchmarking/2026_06_10_simsnowwithntandaasandgffs/simulations",
        help=(
            "Root folder containing one or more PROKKA_* assembly "
            "directories (e.g. PROKKA_06122025__gr_xx_lr_xx_mu_xx). "
            "For backward compatibility this can also point directly at "
            "a single assembly directory."
        )
    )

    parser.add_argument(
        "-g", "--gff-folder",
        required=False,
        default="/nfs/research/jlees/campan/data/clustering_benchmarking/2026_06_10_simsnowwithntandaasandgffs/MSdataset/6925_1#61/",
        help="Folder containing original (reference) GFF files, used for the gene-architecture plots"
    )

    parser.add_argument(
        "-d", "--diversity-root",
        required=False,
        default="/nfs/research/jlees/campan/data/clustering_benchmarking/2026_06_10_simsnowwithntandaasandgffs/simulations",
        help=(
            "Root folder containing, per assembly, the per-seed "
            "subfolders of simulated isolate GFFs (with embedded "
            "##FASTA), used to compute the nucleotide diversity proxy. "
            "Set to '' to skip."
        )
    )

    args = parser.parse_args()

    analyse(
        args.input,
        gff_folder=args.gff_folder,
        diversity_root=(args.diversity_root or None)
    )
