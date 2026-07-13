import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from umap import UMAP


RANDOM_SEED = 34

# Edit these constants directly before running if you want to tune the models.
TSNE_PERPLEXITY = 10
TSNE_MAX_ITER = 3000

UMAP_NEIGHBOURS = 5
UMAP_EPOCHS = 400

HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 5
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.1

# These are only used by the commented alternative clustering methods below.
KMEANS_N_CLUSTERS = 8
AGGLOMERATIVE_N_CLUSTERS = 8
GMM_N_COMPONENTS = 8


def parse_args():
    """Read command-line options that describe input/output paths and execution resources."""
    parser = argparse.ArgumentParser(
        description="Cluster one triangular pairwise distance file."
    )
    parser.add_argument(
        "--dist-file",
        required=True,
        type=Path,
        help="Tab-separated triangular distance file. The distance is read from the last column.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./distance_clustering"),
        help="Directory where models, plots, statistics, and TSV output are written.",
    )
    parser.add_argument(
        "--nthreads",
        "-j",
        type=int,
        default=80,
        help="Number of threads passed to algorithms that support parallel execution.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
        help="Plot formats to write.",
    )
    return parser.parse_args()


def parse_distance_file(path):
    """Load a triangular pairwise distance file into sample names and a full matrix."""
    samples = []
    distances = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(
                    f"Expected at least three tab-separated columns at line {line_number}"
                )

            sample_a = fields[0]
            sample_b = fields[1]
            distance = float(fields[-1])

            for sample in (sample_a, sample_b):
                if sample not in samples:
                    samples.append(sample)

            distances[(sample_a, sample_b)] = distance

    if len(samples) < 2:
        raise ValueError("The distance file must contain at least two samples")

    index = {sample: i for i, sample in enumerate(samples)}
    matrix = np.zeros((len(samples), len(samples)), dtype=float)

    # The original scripts expect triangular pairwise distances. Using sample names
    # to fill the matrix makes the parser independent of the exact row ordering.
    for (sample_a, sample_b), distance in distances.items():
        i = index[sample_a]
        j = index[sample_b]
        matrix[i, j] = distance
        matrix[j, i] = distance

    expected_pairs = len(samples) * (len(samples) - 1) // 2
    if len(distances) != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} unique pairwise distances for {len(samples)} "
            f"samples, found {len(distances)}"
        )

    return samples, matrix


def fit_tsne(matrix, nthreads):
    """Embed the precomputed distance matrix into two dimensions with t-SNE."""
    perplexity = min(TSNE_PERPLEXITY, max(1, matrix.shape[0] - 1))
    model = TSNE(
        metric="precomputed",
        n_jobs=nthreads,
        perplexity=perplexity,
        init="random",
        random_state=RANDOM_SEED,
        verbose=3,
        max_iter=TSNE_MAX_ITER,
    )
    coords = model.fit_transform(matrix)
    return model, coords


def fit_umap(matrix, nthreads):
    """Embed the precomputed distance matrix into two dimensions with UMAP."""
    # UMAP with precomputed distances is useful for visualising the same distance
    # matrix as HDBSCAN sees it. Parallel UMAP can be non-deterministic.
    model = UMAP(
        metric="precomputed",
        n_jobs=nthreads,
        verbose=True,
        n_neighbors=UMAP_NEIGHBOURS,
        n_epochs=UMAP_EPOCHS,
    )
    coords = model.fit_transform(matrix)
    return model, coords


def fit_hdbscan_dist(matrix, nthreads):
    """Cluster samples directly from the precomputed distance matrix."""
    model = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_epsilon=HDBSCAN_CLUSTER_SELECTION_EPSILON,
        allow_single_cluster=True,
        core_dist_n_jobs=nthreads,
        metric="precomputed",
    )
    labels = model.fit_predict(matrix)

    # Alternative for a fixed number of clusters on the distance matrix:
    # model = AgglomerativeClustering(
    #     n_clusters=AGGLOMERATIVE_N_CLUSTERS,
    #     metric="precomputed",
    #     linkage="average",
    # )
    # labels = model.fit_predict(matrix)

    return model, labels


def fit_hdbscan_embedding(coords, nthreads):
    """Cluster samples from two-dimensional embedding coordinates."""
    model = hdbscan.HDBSCAN(
        algorithm="boruvka_balltree",
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_epsilon=HDBSCAN_CLUSTER_SELECTION_EPSILON,
        allow_single_cluster=True,
        core_dist_n_jobs=nthreads,
    )
    labels = model.fit_predict(coords)

    # Alternative for a fixed number of clusters in embedding space:
    # model = KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    # labels = model.fit_predict(coords)

    # Alternative hierarchical clustering in embedding space:
    # model = AgglomerativeClustering(n_clusters=AGGLOMERATIVE_N_CLUSTERS)
    # labels = model.fit_predict(coords)

    # Alternative mixture model in embedding space:
    # model = GaussianMixture(n_components=GMM_N_COMPONENTS, random_state=RANDOM_SEED)
    # labels = model.fit_predict(coords)

    return model, labels


def plot_clusters(coords, labels, output_stem, formats):
    """Draw one scatter plot for a set of coordinates coloured by cluster label."""
    fig = plt.figure(1, dpi=150)
    ax = fig.subplots()
    ax.set_prop_cycle(
        color=[
            "orange",
            "green",
            "red",
            "purple",
            "brown",
            "pink",
            "blue",
            "olive",
            "cyan",
            "lightcoral",
            "darkred",
            "khaki",
            "peru",
            "gold",
            "greenyellow",
            "aquamarine",
            "deepskyblue",
            "fuchsia",
            "hotpink",
        ]
    )

    label_values = sorted(set(labels))
    if label_values and label_values[0] == -1:
        ax.scatter(
            coords[labels == -1, 0],
            coords[labels == -1, 1],
            s=0.15,
            label="Background cluster",
            c="gray",
        )
        label_values = label_values[1:]

    for label in label_values:
        ax.scatter(
            coords[labels == label, 0],
            coords[labels == label, 1],
            s=1,
            label=f"Cluster {label}",
        )

    for fmt in formats:
        fig.savefig(output_stem.with_suffix(f".{fmt}"))
    plt.close(fig)


def save_pickle(path, **data):
    """Store model outputs in the same pickle style as the original scripts."""
    with path.open("wb") as handle:
        pickle.dump(data, handle)


def cluster_counts(labels):
    """Return JSON-friendly cluster label counts."""
    return {str(label): count for label, count in sorted(Counter(labels).items())}


def representative_for_cluster(member_indices, matrix):
    """Choose the most central cluster member by mean within-cluster distance."""
    if len(member_indices) == 1:
        return member_indices[0]

    submatrix = matrix[np.ix_(member_indices, member_indices)]
    mean_distances = submatrix.sum(axis=1) / (len(member_indices) - 1)
    return member_indices[int(np.argmin(mean_distances))]


def write_cdhit_like_tsv(path, samples, matrix, label_sets):
    """Write a member-level clustering table inspired by CD-HIT cluster output."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "method",
                "cluster_id",
                "representative",
                "member",
                "member_index",
                "cluster_size",
                "is_representative",
            ]
        )

        for method, labels in label_sets.items():
            for cluster_id in sorted(set(labels)):
                member_indices = np.where(labels == cluster_id)[0]
                representative_index = representative_for_cluster(member_indices, matrix)
                representative = samples[representative_index]

                for member_index in member_indices:
                    writer.writerow(
                        [
                            method,
                            cluster_id,
                            representative,
                            samples[member_index],
                            member_index,
                            len(member_indices),
                            member_index == representative_index,
                        ]
                    )


def matrix_stats(matrix):
    """Summarise only the non-redundant upper triangle of the distance matrix."""
    upper_triangle = matrix[np.triu_indices_from(matrix, k=1)]
    return {
        "min_distance": float(np.min(upper_triangle)),
        "max_distance": float(np.max(upper_triangle)),
        "nonzero_distances": int(np.count_nonzero(upper_triangle)),
        "nonone_distances": int(np.count_nonzero(upper_triangle != 1.0)),
        "pairwise_distances": int(len(upper_triangle)),
    }


def write_stats(out_dir, samples, matrix, label_sets):
    """Write human-readable and machine-readable summaries of the run."""
    stats = {
        "sample_count": len(samples),
        "matrix": matrix_stats(matrix),
        "clusters": {
            method: {
                "cluster_count": len(set(labels)),
                "label_counts": cluster_counts(labels),
            }
            for method, labels in label_sets.items()
        },
    }

    with (out_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    lines = [
        f"Sample count: {stats['sample_count']}",
        f"Pairwise distances: {stats['matrix']['pairwise_distances']}",
        f"Non-zero distances: {stats['matrix']['nonzero_distances']}",
        f"Non-one distances: {stats['matrix']['nonone_distances']}",
        f"Minimum distance: {stats['matrix']['min_distance']}",
        f"Maximum distance: {stats['matrix']['max_distance']}",
        "",
        "Clusters:",
    ]
    for method, cluster_stats in stats["clusters"].items():
        lines.append(f"- {method}: {cluster_stats['label_counts']}")

    (out_dir / "stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """Run the full load, embedding, clustering, plotting, and reporting workflow."""
    args = parse_args()
    model_dir = args.out_dir / "models"
    plot_dir = args.out_dir / "plots"
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_SEED)
    plt.rcParams.update({"figure.max_open_warning": 0})

    print(f"Loading distances from {args.dist_file}")
    samples, matrix = parse_distance_file(args.dist_file)

    print("Training t-SNE")
    tsne_model, tsne_coords = fit_tsne(matrix, args.nthreads)
    save_pickle(model_dir / "tsne.pk", model=tsne_model, coords=tsne_coords)

    print("Training UMAP")
    umap_model, umap_coords = fit_umap(matrix, args.nthreads)
    save_pickle(model_dir / "umap.pk", model=umap_model, coords=umap_coords)

    print("Clustering precomputed distances")
    hdbs_dist_model, hdbs_dist_labels = fit_hdbscan_dist(matrix, args.nthreads)
    save_pickle(model_dir / "hdbs_dist.pk", model=hdbs_dist_model, labels=hdbs_dist_labels)

    print("Clustering t-SNE coordinates")
    hdbs_tsne_model, hdbs_tsne_labels = fit_hdbscan_embedding(tsne_coords, args.nthreads)
    save_pickle(model_dir / "hdbs_tsne.pk", model=hdbs_tsne_model, labels=hdbs_tsne_labels)

    print("Clustering UMAP coordinates")
    hdbs_umap_model, hdbs_umap_labels = fit_hdbscan_embedding(umap_coords, args.nthreads)
    save_pickle(model_dir / "hdbs_umap.pk", model=hdbs_umap_model, labels=hdbs_umap_labels)

    label_sets = {
        "hdbscan_dist": hdbs_dist_labels,
        "hdbscan_tsne": hdbs_tsne_labels,
        "hdbscan_umap": hdbs_umap_labels,
    }

    print("Writing plots")
    plot_clusters(
        tsne_coords,
        hdbs_dist_labels,
        plot_dir / "cluster_TSNE_HDBSCANdist",
        args.formats,
    )
    plot_clusters(
        tsne_coords,
        hdbs_tsne_labels,
        plot_dir / "cluster_TSNE_HDBSCAN",
        args.formats,
    )
    plot_clusters(
        umap_coords,
        hdbs_dist_labels,
        plot_dir / "cluster_UMAP_HDBSCANdist",
        args.formats,
    )
    plot_clusters(
        umap_coords,
        hdbs_umap_labels,
        plot_dir / "cluster_UMAP_HDBSCAN",
        args.formats,
    )

    print("Writing statistics and cluster TSV")
    write_stats(args.out_dir, samples, matrix, label_sets)
    write_cdhit_like_tsv(args.out_dir / "clusters.tsv", samples, matrix, label_sets)

    print(f"Finished. Outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
