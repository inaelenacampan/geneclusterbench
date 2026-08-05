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
import scipy.sparse as sp
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

# These are now only fallback defaults for fit_hdbscan_dist/fit_hdbscan_embedding
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
    parser.add_argument(
        "--sparse",
        action="store_true",
        help=(
            "Build and cluster a sparse (scipy.sparse CSR) distance matrix "
            "instead of a dense numpy array. Required for real-data runs "
            "where sketchlib's `dist` was restricted with --knn/--neighbourhood, "
            "so the file only contains a subset of pairwise distances. "
            "Missing pairs are treated as 'no measured edge' rather than "
            "filled with a placeholder distance, so no full n x n matrix is "
            "ever built in memory. HDBSCAN and UMAP consume the sparse "
            "matrix directly; t-SNE is attempted on the sparse precomputed "
            "graph and is skipped (not densified) if that isn't supported "
            "for the given data."
        ),
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Stop after the parameter sweep instead of continuing to the full "
            "pipeline. The sweep itself (grid of HDBSCAN min_cluster_size/"
            "min_samples on the precomputed distance matrix, no UMAP/t-SNE, "
            "no plots) now always runs regardless of this flag, since the "
            "full pipeline needs its chosen combination to proceed. Without "
            "this flag, the full pipeline (UMAP/t-SNE, plots, stats, TSV) "
            "runs afterwards using the combination the sweep selected."
        ),
    )
    return parser.parse_args()


def parse_distance_file(path, sparse=False):
    """Load a triangular pairwise distance file into sample names and a matrix.

    With sparse=False (default, unchanged behaviour): builds a dense n x n
    numpy array and requires every one of the n*(n-1)/2 pairs to be present
    (as before). This is fine for small/simulated datasets with a complete
    all-vs-all distance file.

    With sparse=True: builds a scipy.sparse CSR matrix containing only the
    pairs actually present in the file, and does NOT require completeness.
    This is the path for real-data runs where sketchlib's `dist` was run
    with --knn/--neighbourhood, so only a subset of pairwise distances was
    ever computed/written. A missing (a, b) pair means "no distance was
    measured between a and b" -- it is left unstored (no implicit edge),
    never filled with a placeholder value, so memory stays proportional to
    the number of measured pairs rather than n^2.
    """
    samples = []
    sample_index = {}
    rows = []
    cols = []
    data = []
    n_pairs = 0

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
                if sample not in sample_index:
                    sample_index[sample] = len(samples)
                    samples.append(sample)

            i = sample_index[sample_a]
            j = sample_index[sample_b]

            if sparse:
                rows.append(i)
                cols.append(j)
                data.append(distance)
                rows.append(j)
                cols.append(i)
                data.append(distance)
                n_pairs += 1
            # else: dense path re-reads the file below (keeps the original
            # code path/behaviour untouched), so nothing to accumulate here.

    if len(samples) < 2:
        raise ValueError("The distance file must contain at least two samples")

    if sparse:
        n = len(samples)
        matrix = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        # Deliberately do NOT call eliminate_zeros() here. A genuinely-measured
        # zero distance (e.g. two samples with identical/near-identical
        # sketches, which do occur on real data -- duplicate or highly similar
        # gene sequences) must stay distinguishable from a pair whose distance
        # was never measured (no stored entry at all). Collapsing the two by
        # stripping explicit zeros silently disconnects the mutual-reachability
        # graph HDBSCAN builds internally, which is why relative_validity_ can
        # come back NaN for every combination in the sweep. Explicit zero
        # entries cost a little extra memory in the CSR structure but keep the
        # graph connectivity (and therefore relative_validity_) correct.
        return samples, matrix

    # Dense path: rebuild the (sample_a, sample_b) -> distance mapping and
    # keep the original completeness check, unchanged from before.
    distances = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            sample_a, sample_b = fields[0], fields[1]
            distances[(sample_a, sample_b)] = float(fields[-1])

    index = sample_index
    matrix = np.zeros((len(samples), len(samples)), dtype=float)
    for (sample_a, sample_b), distance in distances.items():
        i = index[sample_a]
        j = index[sample_b]
        matrix[i, j] = distance
        matrix[j, i] = distance

    expected_pairs = len(samples) * (len(samples) - 1) // 2
    if len(distances) != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} unique pairwise distances for {len(samples)} "
            f"samples, found {len(distances)}. Pass --sparse if this file was "
            f"generated with a restricted neighbourhood/kNN and is expected "
            f"to be incomplete."
        )

    return samples, matrix


def sweep_hdbscan_params(
    matrix,
    nthreads,
    min_cluster_sizes=SWEEP_MIN_CLUSTER_SIZES,
    min_samples_list=SWEEP_MIN_SAMPLES,
):
    """Try a grid of HDBSCAN params directly on the precomputed distance matrix
    and report cluster counts / noise fraction / internal validity for each
    combination. This skips UMAP/t-SNE and plotting, so it's cheap enough to
    run every time before committing to a final pair of parameters.

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
                metric="precomputed",
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
    """Embed the precomputed distance matrix into two dimensions with t-SNE.
    t-SNE = t-distributed Stochastic Neighbor Embedding

    Works with either a dense numpy array or a scipy.sparse CSR matrix in
    `matrix` -- sklearn's TSNE(metric="precomputed") accepts a sparse CSR
    precomputed *graph* (i.e. only the stored/measured distances, as produced
    by --sparse) and never requires it to be densified first. Each sample
    needs at least ~3*perplexity stored neighbours for this to work; with a
    sketchlib --knn/--neighbourhood value in the hundreds/thousands and the
    perplexity used here, that's comfortably satisfied.

    If fitting on a sparse matrix fails for a genuine data reason (e.g. some
    samples end up with too few stored neighbours, or the neighbour graph is
    disconnected in a way sklearn can't route through), this does NOT fall
    back to a dense computation -- per policy, t-SNE is simply skipped for
    this run. Returns (None, None) in that case; callers must check for this.
    """
    is_sparse = sp.issparse(matrix)
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
    if not is_sparse:
        coords = model.fit_transform(matrix)
        return model, coords

    try:
        coords = model.fit_transform(matrix)
        return model, coords
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        print(
            "t-SNE on the sparse precomputed distance matrix failed "
            f"({type(exc).__name__}: {exc}). Skipping t-SNE for this run "
            "rather than densifying the matrix."
        )
        return None, None


def fit_umap(matrix, nthreads):
    """Embed the precomputed distance matrix into two dimensions with UMAP.

    Accepts either a dense numpy array or a scipy.sparse CSR matrix. UMAP's
    metric="precomputed" natively supports a sparse precomputed distance
    graph (only stored/measured entries are treated as neighbours) and never
    requires densifying it first.
    """
    # umap = Uniform Manifold Approximation and Projection
    # UMAP with precomputed distances is useful for visualising the same distance
    # matrix as HDBSCAN sees it. Parallel UMAP can be non-deterministic.
    model = UMAP(
        metric="precomputed",
        n_jobs=nthreads,
        verbose=True,
        n_neighbors=UMAP_NEIGHBOURS,
        min_dist=UMAP_MIN_DIST,
        n_epochs=UMAP_EPOCHS,
        force_approximation_algorithm=sp.issparse(matrix),
    )
    coords = model.fit_transform(matrix)
    return model, coords


def fit_hdbscan_dist(
    matrix,
    nthreads,
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
):
    """Cluster samples directly from the precomputed distance matrix."""
    # Hierarchical Density-Based Spatial Clustering of Applications with Noise
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
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


def fit_hdbscan_embedding(
    coords,
    nthreads,
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
):
    """Cluster samples from two-dimensional embedding coordinates."""
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
    """Choose the most central cluster member by mean within-cluster distance."""
    if len(member_indices) == 1:
        return member_indices[0]

    if sp.issparse(matrix):
        submatrix = matrix[member_indices, :][:, member_indices]
        # With a sparse matrix, an unstored entry means "distance not
        # measured" rather than "distance 0" -- dividing by the full
        # (cluster_size - 1) would silently treat missing neighbours as
        # perfectly identical and bias the representative towards members
        # with the fewest measured distances. Instead, average only over
        # the distances that were actually measured for each member.
        row_sums = np.asarray(submatrix.sum(axis=1)).ravel()
        row_counts = np.diff(submatrix.indptr)
        row_counts = np.where(row_counts == 0, 1, row_counts)  # avoid /0
        mean_distances = row_sums / row_counts
    else:
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
    (dist-only vs. embedding+HDBSCAN combined) are both written, so the file
    directly answers "how long did method X take" as well as "where did the
    time go within method X".
    """
    lines = []
    for label, seconds in timings:
        lines.append(f"{label}: {seconds:.3f}s")
    (out_dir / "time_per_method.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def matrix_stats(matrix):
    """Summarise only the non-redundant upper triangle of the distance matrix.

    For a sparse matrix, "pairwise_distances" is the count of pairs that
    actually had a measured/stored distance (which will be far smaller than
    n*(n-1)/2 when built from a --knn/--neighbourhood-restricted file), not
    the full theoretical pair count.
    """
    if sp.issparse(matrix):
        coo = matrix.tocoo()
        mask = coo.row < coo.col  # upper triangle only, matrix is symmetric
        upper_triangle = coo.data[mask]
        if upper_triangle.size == 0:
            return {
                "min_distance": None,
                "max_distance": None,
                "nonzero_distances": 0,
                "nonone_distances": 0,
                "pairwise_distances": 0,
            }
        return {
            "min_distance": float(np.min(upper_triangle)),
            "max_distance": float(np.max(upper_triangle)),
            "nonzero_distances": int(np.count_nonzero(upper_triangle)),
            "nonone_distances": int(np.count_nonzero(upper_triangle != 1.0)),
            "pairwise_distances": int(upper_triangle.size),
        }

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
    """Run the full load, sweep, embedding, clustering, plotting, and
    reporting workflow.

    The parameter sweep now always runs (not just with --sweep): it's cheap
    relative to the rest of the pipeline, and the full pipeline needs a
    (min_cluster_size, min_samples) pair to run with anyway. --sweep is kept
    as a flag to stop after the sweep, e.g. for inspecting sweep.json by hand
    before trusting the automatic choice.

    Timing: every stage is timed individually and written to time.txt. This
    is a breakdown *within* this script only (loading, sweep, embeddings,
    each HDBSCAN fit) -- it does not include the upstream `sketch`/`dist`
    steps run by the submit script, which are timed separately via
    timebenchmark.txt in the SLURM scaffold. To get "sketch + method" time,
    add the relevant time.txt entry here to the sketch+dist portion of that
    external timing.
    """
    args = parse_args()
    run_start = time.time()
    timings = []

    np.random.seed(RANDOM_SEED)
    plt.rcParams.update({"figure.max_open_warning": 0})

    print(f"Loading distances from {args.dist_file}" + (" (sparse)" if args.sparse else ""))
    t0 = time.time()
    samples, matrix = parse_distance_file(args.dist_file, sparse=args.sparse)
    timings.append(("load_distance_matrix", time.time() - t0))

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
    tsne_available = tsne_coords is not None
    if tsne_available:
        save_pickle(model_dir / "tsne.pk", model=tsne_model, coords=tsne_coords)
    else:
        print("t-SNE skipped (see message above); no tsne.pk written, "
              "t-SNE plots/clustering will be omitted from this run.")

    print("Training UMAP")
    t0 = time.time()
    umap_model, umap_coords = fit_umap(matrix, args.nthreads)
    umap_embed_time = time.time() - t0
    timings.append(("umap_embedding", umap_embed_time))
    save_pickle(model_dir / "umap.pk", model=umap_model, coords=umap_coords)

    print("Clustering precomputed distances")
    t0 = time.time()
    hdbs_dist_model, hdbs_dist_labels = fit_hdbscan_dist(
        matrix, args.nthreads, min_cluster_size, min_samples
    )
    hdbs_dist_time = time.time() - t0
    timings.append(("hdbscan_dist_fit", hdbs_dist_time))
    save_pickle(model_dir / "hdbs_dist.pk", model=hdbs_dist_model, labels=hdbs_dist_labels)

    if tsne_available:
        print("Clustering t-SNE coordinates")
        t0 = time.time()
        hdbs_tsne_model, hdbs_tsne_labels = fit_hdbscan_embedding(
            tsne_coords, args.nthreads, min_cluster_size, min_samples
        )
        hdbs_tsne_time = time.time() - t0
        timings.append(("hdbscan_tsne_fit", hdbs_tsne_time))
        save_pickle(model_dir / "hdbs_tsne.pk", model=hdbs_tsne_model, labels=hdbs_tsne_labels)
    else:
        hdbs_tsne_model, hdbs_tsne_labels = None, None
        hdbs_tsne_time = 0.0

    print("Clustering UMAP coordinates")
    t0 = time.time()
    hdbs_umap_model, hdbs_umap_labels = fit_hdbscan_embedding(
        umap_coords, args.nthreads, min_cluster_size, min_samples
    )
    hdbs_umap_time = time.time() - t0
    timings.append(("hdbscan_umap_fit", hdbs_umap_time))
    save_pickle(model_dir / "hdbs_umap.pk", model=hdbs_umap_model, labels=hdbs_umap_labels)

    # Per-method totals: "method" here means one of the three ways of getting
    # from the distance matrix to a final clustering. hdbscan_dist has no
    # embedding step, so its total is just its own fit time; the other two
    # methods pay for their embedding as well as their HDBSCAN fit.
    timings.append(("method_hdbscan_dist_total", hdbs_dist_time))
    if tsne_available:
        timings.append(("method_hdbscan_tsne_total", tsne_embed_time + hdbs_tsne_time))
    timings.append(("method_hdbscan_umap_total", umap_embed_time + hdbs_umap_time))

    label_sets = {
        "hdbscan_dist": hdbs_dist_labels,
        "hdbscan_umap": hdbs_umap_labels,
    }
    if tsne_available:
        label_sets["hdbscan_tsne"] = hdbs_tsne_labels

    print("Writing plots")
    t0 = time.time()
    if tsne_available:
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
