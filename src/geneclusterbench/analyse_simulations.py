#!/usr/bin/env python3

import os
import glob
import argparse
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA


# ---------------------------------------------------------
# FILE HANDLING
# ---------------------------------------------------------

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

def plot_rarefaction_cloud(accumulations, out):

    arr = np.array(accumulations)

    mean = arr.mean(axis=0)
    sd = arr.std(axis=0)

    x = np.arange(1, len(mean) + 1)

    plt.figure(figsize=(8, 6))

    for curve in arr:
        plt.plot(x, curve, alpha=0.20)

    plt.plot(
        x,
        mean,
        color="black",
        linewidth=3
    )

    plt.fill_between(
        x,
        mean - sd,
        mean + sd,
        alpha=0.3
    )

    plt.xlabel("Number of isolates")
    plt.ylabel("Pangenome size")
    plt.title("Pangenome rarefaction")

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

def plot_core_decay(core_curves, out):

    arr = np.array(core_curves)

    mean = arr.mean(axis=0)
    sd = arr.std(axis=0)

    x = np.arange(1, len(mean) + 1)

    plt.figure(figsize=(8, 6))

    for curve in arr:
        plt.plot(x, curve, alpha=0.2)

    plt.plot(
        x,
        mean,
        color="black",
        lw=3
    )

    plt.fill_between(
        x,
        mean - sd,
        mean + sd,
        alpha=0.3
    )

    plt.xlabel("Number of isolates")
    plt.ylabel("Core genes")
    plt.title("Core genome decay")

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


def plot_openness(accumulations, out):

    alphas = estimate_heaps(accumulations)

    plt.figure(figsize=(6, 6))

    sns.violinplot(y=alphas)

    plt.ylabel("Heaps alpha")
    plt.title("Pangenome openness")

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
    gff_folder=None
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

    for f in files:

        print("Loading:", f)

        mat = load_pa(f)

        mats.append(mat)

        accumulations.append(
            accumulation(mat)
        )

        cores.append(
            core_decay(mat)
        )

        pangenome_sizes.append(
            len(mat)
        )

    representative = mats[0]

    plot_rarefaction_cloud(
        accumulations,
        out
    )

    plot_core_decay(
        cores,
        out
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
        out
    )

    plot_phase_space(
        mats,
        out
    )

    pd.DataFrame({
        "pangenome_size":
            pangenome_sizes
    }).to_csv(
        os.path.join(
            folder,
            "summary_statistics.csv"
        ),
        index=False
    )
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
        help="Folder containing original GFF files"
    )

    args = parser.parse_args()

    analyse(
        args.input,
        args.gff_folder
    )
