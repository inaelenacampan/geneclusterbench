import argparse
import csv
import json
import pickle
import time
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
UMAP_MIN_DIST = 0.0  # lower = tighter packing in embedding space, easier to find many small clusters

# These are now only fallback defaults for fit_hdbscan_raw/fit_hdbscan_embedding
# when called without explicit values (e.g. from other code). main() no longer
# reads these directly -- it uses whatever (min_cluster_size, min_samples) the
# sweep's choose_best_combination() selects, every run.
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 2
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.1

# These are only used by the commented alternative clustering methods below.
KMEANS_N_CLUSTERS = 8
AGGLOMERATIVE_N_CLUSTERS = 8
GMM_N_COMPONENTS = 8

# Grid used by --sweep. Edit these directly to widen/narrow the search.
SWEEP_MIN_CLUSTER_SIZES = (2, 3, 5, 8, 12)
SWEEP_MIN_SAMPLES = (1, 2, 3, 5)

# Combos with min_cluster_size above this are excluded from the "best combo"
# selection: real gene families are frequently small (near-singleton), so a
# large min_cluster_size systematically throws away genuine small families as
# noise rather than reflecting anything about clustering quality. They are
# still run and reported in the sweep output for transparency, just not
# eligible to be picked.
SWEEP_MAX_ELIGIBLE_MIN_CLUSTER_SIZE = 8

# ProstT5 mean-pooled embeddings are dense feature vectors, NOT a precomputed
# pairwise distance matrix. Every model below is therefore run with
# metric="euclidean" (or left at its raw-feature default) on the embedding
# matrix directly, unlike cluster_distance_file.py which passes
# metric="precomputed" to everything. Mixing these two up silently produces
# nonsense: HDBSCAN/TSNE/UMAP would try to interpret embedding *values* as if
# they were an n x n distance matrix.
EMBEDDING_METRIC = "cosine"


def parse_args():
    """Read command-line options that describe input/output paths and execution resources."""
    parser = argparse.ArgumentParser(
        description="Cluster one ProstT5 protein embedding pickle."
    )
    parser.add_argument(
        "--embeddings-file",
        required=True,
        type=Path,
        help="Pickle produced by embedprots.py, containing 'ids' and 'embd' (n_samples x n_dims).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./embeddings_clustering"),
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
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Stop after the parameter sweep instead of continuing to the full "
            "pipeline. The sweep itself (grid of HDBSCAN min_cluster_size/"
            "min_samples on the raw embedding matrix, no UMAP/t-SNE, "
            "no plots) now always runs regardless of this flag, since the "
            "full pipeline needs its chosen combination to proceed. Without "
            "this flag, the full pipeline (UMAP/t-SNE, plots, stats, TSV) "
            "runs afterwards using the combination the sweep selected."
        ),
    )
    return parser.parse_args()


def load_embeddings(path):
    """Load the {ids, prots, embd} pickle written by embedprots.py.

    Returns the sample names and the raw (n_samples x n_dims) embedding
    matrix -- this is a feature matrix, not a pairwise distance matrix.
    """
    with path.open("rb") as handle:
        data = pickle.load(handle)

    samples = list(data["ids"])
    matrix = np.asarray(data["embd"], dtype=float)

    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D (n_samples x n_dims) embedding matrix, got shape {matrix.shape}"
        )
    if matrix.shape[0] != len(samples):
        raise ValueError(
            f"Embedding matrix has {matrix.shape[0]} rows but there are "
            f"{len(samples)} ids"
        )
    if len(samples) < 2:
        raise ValueError("The embeddings file must contain at least two samples")

    return samples, matrix


def sweep_hdbscan_params(
    matrix,
    nthreads,
    min_cluster_sizes=SWEEP_MIN_CLUSTER_SIZES,
    min_samples_list=SWEEP_MIN_SAMPLES,
):
    """Try a grid of HDBSCAN params directly on the raw embedding matrix
    (euclidean metric) and report cluster counts / noise fraction / internal
    validity for each combination. This skips UMAP/t-SNE and plotting, so
    it's cheap enough to run every time before committing to a final pair of
    parameters.

    relative_validity_ is HDBSCAN's built-in, fast approximation of DBCV
    (Density-Based Clustering Validation). It needs no ground truth, so it's
    usable both here (where we happen to have simulated ground truth to
    sanity-check against) and later on real data (where we won't). Higher is
    better. cluster_persistence_ is also logged (mean across clusters) as a
    secondary stability signal: it reflects how long each cluster survives
    across the density hierarchy rather than being an artifact of one cut.
    """
    results = []
    print(
        f"{'min_cluster_size':>17} {'min_samples':>12} {'n_clusters':>11} "
        f"{'noise_frac':>11} {'rel_validity':>13} {'mean_persist':>13}"
    )
    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            model = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                cluster_selection_epsilon=HDBSCAN_CLUSTER_SELECTION_EPSILON,
                allow_single_cluster=True,
                core_dist_n_jobs=nthreads,
                metric=EMBEDDING_METRIC,
                gen_min_span_tree=True,  # required for relative_validity_
            )
            labels = model.fit_predict(matrix)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_frac = float(np.mean(labels == -1))

            # relative_validity_ is only meaningful with >=2 real clusters;
            # HDBSCAN can still return it in degenerate cases (e.g. a single
            # cluster), so guard against NaN/errors defensively.
            try:
                relative_validity = float(model.relative_validity_)
            except (AttributeError, ValueError):
                relative_validity = float("nan")

            if model.cluster_persistence_.size:
                mean_persistence = float(np.mean(model.cluster_persistence_))
            else:
                mean_persistence = float("nan")

            results.append(
                {
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "n_clusters": n_clusters,
                    "noise_fraction": round(noise_frac, 3),
                    "relative_validity": round(relative_validity, 4)
                    if relative_validity == relative_validity  # NaN check
                    else None,
                    "mean_cluster_persistence": round(mean_persistence, 4)
                    if mean_persistence == mean_persistence
                    else None,
                }
            )
            print(
                f"{mcs:>17} {ms:>12} {n_clusters:>11} {noise_frac:>10.1%} "
                f"{relative_validity:>13.4f} {mean_persistence:>13.4f}"
            )
    return results


def choose_best_combination(
    results, max_eligible_min_cluster_size=SWEEP_MAX_ELIGIBLE_MIN_CLUSTER_SIZE
):
    """Pick the (min_cluster_size, min_samples) combo to actually cluster with.

    Selection is purely intrinsic (no ground truth needed, so this works the
    same way on real data as on simulated data):
      1. Rule out combos with min_cluster_size above the eligible threshold
         (biology: real gene families are often small, so large thresholds
         just discard genuine small families as noise).
      2. Among the survivors, pick the highest relative_validity_ (HDBSCAN's
         internal DBCV approximation). Ties are broken by higher mean
         cluster persistence, then by lower noise fraction.
    """
    eligible = [
        r
        for r in results
        if r["min_cluster_size"] <= max_eligible_min_cluster_size
        and r["relative_validity"] is not None
    ]
    if not eligible:
        raise RuntimeError(
            "No eligible HDBSCAN parameter combination found in the sweep "
            f"(min_cluster_size <= {max_eligible_min_cluster_size} with a "
            "valid relative_validity_ score)."
        )

    best = max(
        eligible,
        key=lambda r: (
            r["relative_validity"],
            r["mean_cluster_persistence"] or float("-inf"),
            -r["noise_fraction"],
        ),
    )
    return best


def fit_tsne(matrix, nthreads):
    """Embed the raw protein-embedding matrix into two dimensions with t-SNE.
    t-SNE = t-distributed Stochastic Neighbor Embedding.

    Unlike cluster_distance_file.py's fit_tsne, this does NOT pass
    metric="precomputed": `matrix` here is n_samples x n_dims raw features,
    so t-SNE computes its own euclidean distances internally. init="pca" is
    used (rather than "random") since it tends to give a more stable layout
    for dense feature vectors like these.
    """
    perplexity = min(TSNE_PERPLEXITY, max(1, matrix.shape[0] - 1))
    model = TSNE(
        metric=EMBEDDING_METRIC,
        n_jobs=nthreads,
        perplexity=perplexity,
        init="pca",
        random_state=RANDOM_SEED,
        verbose=3,
        max_iter=TSNE_MAX_ITER,
    )
    coords = model.fit_transform(matrix)
    return model, coords


def fit_umap(matrix, nthreads):
    """Embed the raw protein-embedding matrix into two dimensions with UMAP.

    Unlike cluster_distance_file.py's fit_umap, this does NOT pass
    metric="precomputed": `matrix` here is n_samples x n_dims raw features.
    """
    # umap = Uniform Manifold Approximation and Projection
    model = UMAP(
        metric=EMBEDDING_METRIC,
        n_jobs=nthreads,
        verbose=True,
        n_neighbors=UMAP_NEIGHBOURS,
        min_dist=UMAP_MIN_DIST,
        n_epochs=UMAP_EPOCHS,
        random_state=RANDOM_SEED,
    )
    coords = model.fit_transform(matrix)
    return model, coords


def fit_hdbscan_raw(
    matrix,
    nthreads,
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
):
    """Cluster samples directly from the raw embedding matrix (euclidean),
    analogous to fit_hdbscan_dist() in cluster_distance_file.py but without
    metric="precomputed", since `matrix` holds feature vectors, not
    pairwise distances.
    """
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=HDBSCAN_CLUSTER_SELECTION_EPSILON,
        allow_single_cluster=True,
        core_dist_n_jobs=nthreads,
        metric=EMBEDDING_METRIC,
    )
    labels = model.fit_predict(matrix)

    # Alternative for a fixed number of clusters on the embedding matrix:
    # model = AgglomerativeClustering(
    #     n_clusters=AGGLOMERATIVE_N_CLUSTERS,
    #     metric=EMBEDDING_METRIC,
    #     linkage="average",
    # )
    # labels = model.fit_predict(matrix)

    return model, labels


def fit_hdbscan_embedding(
    coords,
    nthreads,
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
):
    """Cluster samples from two-dimensional t-SNE/UMAP embedding coordinates."""
    model = hdbscan.HDBSCAN(
        algorithm="boruvka_balltree",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
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

    label_values = sorted(set(labels))
    n_real_clusters = len(label_values) - (1 if label_values and label_values[0] == -1 else 0)

    # tab20 + tab20b + tab20c gives 60 visually distinct colours, which scales
    # much better than a fixed 19-colour list once you have "a lot" of clusters.
    cmap_names = ["tab20", "tab20b", "tab20c"]
    palette = []
    for name in cmap_names:
        palette.extend([plt.get_cmap(name)(i) for i in range(20)])
    ax.set_prop_cycle(color=palette[: max(n_real_clusters, 1)])

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

    # Skip drawing a legend when there are too many clusters for it to be
    # readable; it would otherwise dominate the figure.
    if n_real_clusters <= 30:
        ax.legend(markerscale=10, fontsize=6, loc="best")

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
    """Choose the cluster member closest to the cluster centroid.

    `matrix` holds raw embedding vectors (n_samples x n_dims), not a
    precomputed distance matrix, so the representative is picked by nearest
    euclidean distance to the mean embedding of the cluster, rather than by
    the lowest mean pairwise distance used in cluster_distance_file.py.
    """
    if len(member_indices) == 1:
        return member_indices[0]

    sub = matrix[member_indices]
    centroid = sub.mean(axis=0)
    distances_to_centroid = np.linalg.norm(sub - centroid, axis=1)
    return member_indices[int(np.argmin(distances_to_centroid))]


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
            # Highest existing cluster ID (ignoring -1)
            next_cluster_id = max([x for x in set(labels) if x != -1], default=-1) + 1

            for cluster_id in sorted(set(labels)):

                member_indices = np.where(labels == cluster_id)[0]

                if cluster_id == -1:
                    # Give every noise point its own singleton cluster
                    for member_index in member_indices:
                        writer.writerow(
                            [
                                method,
                                next_cluster_id,
                                samples[member_index],      # representative
                                samples[member_index],      # member
                                member_index,
                                1,                          # cluster size
                                True,                       # representative
                            ]
                        )
                        next_cluster_id += 1

                else:
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


def write_timings(out_dir, timings):
    """Write a per-stage timing breakdown to time.txt.

    `timings` is an ordered dict/list of (label, seconds) pairs. Stage timings
    (sweep, embeddings, individual HDBSCAN fits) and per-method totals
    (raw-only vs. 2D-embedding+HDBSCAN combined) are both written, so the
    file directly answers "how long did method X take" as well as "where did
    the time go within method X".
    """
    lines = []
    for label, seconds in timings:
        lines.append(f"{label}: {seconds:.3f}s")
    (out_dir / "time_per_method.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def matrix_stats(matrix):
    """Summarise the raw embedding matrix (n_samples x n_dims feature
    vectors). This is deliberately different from cluster_distance_file.py's
    matrix_stats(), which summarises the upper triangle of a pairwise
    distance matrix -- that concept doesn't apply to a raw feature matrix.
    """
    return {
        "n_samples": int(matrix.shape[0]),
        "n_dims": int(matrix.shape[1]),
        "value_min": float(np.min(matrix)),
        "value_max": float(np.max(matrix)),
        "value_mean": float(np.mean(matrix)),
        "value_std": float(np.std(matrix)),
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
        f"Embedding dimensions: {stats['matrix']['n_dims']}",
        f"Value min: {stats['matrix']['value_min']}",
        f"Value max: {stats['matrix']['value_max']}",
        f"Value mean: {stats['matrix']['value_mean']}",
        f"Value std: {stats['matrix']['value_std']}",
        "",
        "Clusters:",
    ]
    for method, cluster_stats in stats["clusters"].items():
        lines.append(f"- {method}: {cluster_stats['label_counts']}")

    (out_dir / "stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """Run the full load, sweep, embedding, clustering, plotting, and
    reporting workflow on ProstT5 protein embeddings.

    This mirrors cluster_distance_file.py's main(), but starts from a raw
    (n_samples x n_dims) embedding matrix (produced by embedprots.py) instead
    of a triangular pairwise distance file. Three clustering methods are
    produced, same as the sketching pipeline:
      - hdbscan_raw:   HDBSCAN directly on the embedding vectors (euclidean)
      - hdbscan_tsne:  HDBSCAN on a 2D t-SNE projection of the embeddings
      - hdbscan_umap:  HDBSCAN on a 2D UMAP projection of the embeddings

    The parameter sweep now always runs (not just with --sweep): it's cheap
    relative to the rest of the pipeline, and the full pipeline needs a
    (min_cluster_size, min_samples) pair to run with anyway. --sweep is kept
    as a flag to stop after the sweep, e.g. for inspecting sweep.json by hand
    before trusting the automatic choice.

    Timing: every stage is timed individually and written to time.txt. This
    is a breakdown *within* this script only (loading, sweep, embeddings,
    each HDBSCAN fit) -- it does not include the upstream embedprots.py
    step run by the submit script, which is timed separately via
    timebenchmark.txt in the SLURM scaffold.
    """
    args = parse_args()
    run_start = time.time()
    timings = []

    np.random.seed(RANDOM_SEED)
    plt.rcParams.update({"figure.max_open_warning": 0})

    print(f"Loading embeddings from {args.embeddings_file}")
    t0 = time.time()
    samples, matrix = load_embeddings(args.embeddings_file)
    timings.append(("load_embeddings", time.time() - t0))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweeping HDBSCAN parameters on {len(samples)} samples")
    t0 = time.time()
    sweep_results = sweep_hdbscan_params(matrix, args.nthreads)
    best_combo = choose_best_combination(sweep_results)
    timings.append(("sweep_total", time.time() - t0))
    print(
        "Chosen combination: min_cluster_size="
        f"{best_combo['min_cluster_size']}, min_samples={best_combo['min_samples']} "
        f"(relative_validity={best_combo['relative_validity']}, "
        f"noise_fraction={best_combo['noise_fraction']})"
    )

    sweep_output = {
        "sweep": sweep_results,
        "chosen_combination": {
            "min_cluster_size": best_combo["min_cluster_size"],
            "min_samples": best_combo["min_samples"],
            "selection_metric": "relative_validity",
            "relative_validity": best_combo["relative_validity"],
            "mean_cluster_persistence": best_combo["mean_cluster_persistence"],
            "noise_fraction": best_combo["noise_fraction"],
            "n_clusters": best_combo["n_clusters"],
            "max_eligible_min_cluster_size": SWEEP_MAX_ELIGIBLE_MIN_CLUSTER_SIZE,
        },
    }
    with (args.out_dir / "sweep.json").open("w", encoding="utf-8") as handle:
        json.dump(sweep_output, handle, indent=2)
    print(f"Sweep results written to {args.out_dir / 'sweep.json'}")

    if args.sweep:
        timings.append(("total", time.time() - run_start))
        write_timings(args.out_dir, timings)
        return

    min_cluster_size = best_combo["min_cluster_size"]
    min_samples = best_combo["min_samples"]

    model_dir = args.out_dir / "models"
    plot_dir = args.out_dir / "plots"
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("Training t-SNE")
    t0 = time.time()
    tsne_model, tsne_coords = fit_tsne(matrix, args.nthreads)
    tsne_embed_time = time.time() - t0
    timings.append(("tsne_embedding", tsne_embed_time))
    save_pickle(model_dir / "tsne.pk", model=tsne_model, coords=tsne_coords)

    print("Training UMAP")
    t0 = time.time()
    umap_model, umap_coords = fit_umap(matrix, args.nthreads)
    umap_embed_time = time.time() - t0
    timings.append(("umap_embedding", umap_embed_time))
    save_pickle(model_dir / "umap.pk", model=umap_model, coords=umap_coords)

    print("Clustering raw embeddings")
    t0 = time.time()
    hdbs_raw_model, hdbs_raw_labels = fit_hdbscan_raw(
        matrix, args.nthreads, min_cluster_size, min_samples
    )
    hdbs_raw_time = time.time() - t0
    timings.append(("hdbscan_raw_fit", hdbs_raw_time))
    save_pickle(model_dir / "hdbs_raw.pk", model=hdbs_raw_model, labels=hdbs_raw_labels)

    print("Clustering t-SNE coordinates")
    t0 = time.time()
    hdbs_tsne_model, hdbs_tsne_labels = fit_hdbscan_embedding(
        tsne_coords, args.nthreads, min_cluster_size, min_samples
    )
    hdbs_tsne_time = time.time() - t0
    timings.append(("hdbscan_tsne_fit", hdbs_tsne_time))
    save_pickle(model_dir / "hdbs_tsne.pk", model=hdbs_tsne_model, labels=hdbs_tsne_labels)

    print("Clustering UMAP coordinates")
    t0 = time.time()
    hdbs_umap_model, hdbs_umap_labels = fit_hdbscan_embedding(
        umap_coords, args.nthreads, min_cluster_size, min_samples
    )
    hdbs_umap_time = time.time() - t0
    timings.append(("hdbscan_umap_fit", hdbs_umap_time))
    save_pickle(model_dir / "hdbs_umap.pk", model=hdbs_umap_model, labels=hdbs_umap_labels)

    # Per-method totals: "method" here means one of the three ways of getting
    # from the embedding matrix to a final clustering. hdbscan_raw has no
    # 2D-projection step, so its total is just its own fit time; the other
    # two methods pay for their embedding as well as their HDBSCAN fit.
    timings.append(("method_hdbscan_raw_total", hdbs_raw_time))
    timings.append(("method_hdbscan_tsne_total", tsne_embed_time + hdbs_tsne_time))
    timings.append(("method_hdbscan_umap_total", umap_embed_time + hdbs_umap_time))

    label_sets = {
        "hdbscan_raw": hdbs_raw_labels,
        "hdbscan_tsne": hdbs_tsne_labels,
        "hdbscan_umap": hdbs_umap_labels,
    }

    print("Writing plots")
    t0 = time.time()
    plot_clusters(
        tsne_coords,
        hdbs_raw_labels,
        plot_dir / "cluster_TSNE_HDBSCANraw",
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
        hdbs_raw_labels,
        plot_dir / "cluster_UMAP_HDBSCANraw",
        args.formats,
    )
    plot_clusters(
        umap_coords,
        hdbs_umap_labels,
        plot_dir / "cluster_UMAP_HDBSCAN",
        args.formats,
    )
    timings.append(("plotting", time.time() - t0))

    print("Writing statistics and cluster TSV")
    t0 = time.time()
    write_stats(args.out_dir, samples, matrix, label_sets)
    write_cdhit_like_tsv(args.out_dir / "clusters.tsv", samples, matrix, label_sets)
    timings.append(("stats_and_tsv", time.time() - t0))

    timings.append(("total", time.time() - run_start))
    write_timings(args.out_dir, timings)

    print(f"Finished. Outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
