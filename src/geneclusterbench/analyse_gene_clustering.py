# import libraries
from sklearn.preprocessing import LabelEncoder
import argparse
import os
import warnings
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import AutoMinorLocator
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_mutual_info_score
from numpy.random import default_rng
import re
import subprocess
import copy
import zlib
from io import StringIO
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Figure formats written by save_figure for every plot in the pipeline,
# and the per-format output subdirectories under each run's --out-folder
# (requirement 6: group figures by format -- png/, pdf/, svg/ -- instead
# of mixing all three extensions together in one flat folder).
FIGURE_FORMATS = ["png", "pdf", "svg"]

DEFAULT_DATAPATH = (
    "/nfs/research/jlees/campan/data/clustering_benchmarking/"
    "2026_06_10_simsnowwithntandaasandgffs"
)
DEFAULT_FONT_REGULAR = (
    "/nfs/research/jlees/campan/data/ibm-plex-sans/fonts/complete/ttf/"
    "IBMPlexSans-Regular.ttf"
)
DEFAULT_FONT_ITALIC = (
    "/nfs/research/jlees/campan/data/ibm-plex-sans/fonts/complete/ttf/"
    "IBMPlexSans-Italic.ttf"
)
DEFAULT_FONT_BOLD = (
    "/nfs/research/jlees/campan/data/ibm-plex-sans/fonts/complete/ttf/"
    "IBMPlexSans-Bold.ttf"
)
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_SEEDS = str(PROJECT_ROOT / "data" / "random_numbers.txt")

CLUSTERERS = [
            "cdhit", 
            "mmseqs2", 
            "diamond", 
            "panaroo",
            "ppanggolin",
            "panta",
            "panx",
            "sketch",
            "embeddings"
            ]
# === (real-data support) ===
# Real biological data has no ground truth and gene IDs are not in the
# "geneid_N" simulation format, so of the original CLUSTERERS list above we
# restrict real-data analysis to only the methods that were explicitly
# requested: CD-HIT, DIAMOND, MMseqs2, Panaroo, Ppanggolin, Panta, PanX, and
# sketch-based clustering. This list is only consumed by the new *_realdata
# functions below; it never touches the simulations code path.
#
# Within "sketch", real data now clusters via UMAP+HDBSCAN and a
# single-linkage connected-components threshold sweep (see
# CONNECTED_COMPONENTS_METHOD_NAMES above) -- t-SNE-based and direct
# distance-based ("hdbscan_dist") clustering are no longer produced for real
# data, since both performed poorly on it. get_dfs_from_sketch_realdata is
# generic over whatever sub-method names actually appear in clusters.tsv, so
# no change is needed there.
REAL_DATA_CLUSTERERS = [
                        "cdhit", 
                        "mmseqs2", 
                        "diamond", 
                        "panaroo", 
                        "ppanggolin", 
                        "panta", 
                        "panx", 
                        "sketch"
                        ]

SEQTYPES = ["nt", "aa"]
PARAMORDER = ["st", "c"]
DEFAULT_PARAMS = {"st": "nt", "c": 0.9}
AXIS_TITLE_FONT_SIZE = 10
BASE_FONT_SIZE = 7
NJ_TREE_TIP_FONT_SIZE = 9  # tip-label font size for dendrogram plots (slightly larger than BASE_FONT_SIZE)
DENDROGRAM_LINKAGE_METHOD = "average"  # UPGMA -- standard choice for clustering a precomputed distance matrix

# Number of label-shuffles used by permutation_test_agreement when computing
# the ARI/AMI-vs-ground-truth p-values reported in outdf (see
# calculate_values_from_cluster_matrix). 1000 gives p-value resolution of
# ~1e-3 (finest reportable value is 1/(nperm+1)) at a runtime cost of
# ~1000 extra ARI/AMI evaluations per (assembly, seed, clusterer) row;
# raise it (e.g. to 10000, permutation_test_agreement's own default) if
# finer-grained p-values are needed and the extra runtime is acceptable.
AGREEMENT_PVALUE_NPERM = 1000
SIGNIFICANCE_ALPHA = 0.05  # threshold used to flag "significant" p-values in plots/tables
ADJ_RAND_INDEX_PVALUE_COL = "adj_rand_index_pvalue"
ADJ_MUTUAL_INFO_PVALUE_COL = "adj_mutual_info_pvalue"
DOPREM = True

SIM_ONLY_SKETCH_METHOD_NAMES = ["hdbscan_dist", "hdbscan_tsne"]

# For real data, cluster_distance_file.py no longer runs hdbscan_dist or
# hdbscan_tsne (both perform poorly on real data, per the pipeline change);
# it replaces them with single-linkage connected-components clustering swept
# over several thresholds, run directly on the sparse nearest-neighbour
# distance graph via networkx. hdbscan_umap is unaffected and still produced
# for both simulated and real data. These threshold values/label names must
# match cluster_distance_file.CONNECTED_COMPONENTS_THRESHOLDS and its
# f"connected_components_t{threshold}" label naming exactly.
#
# === (unification pass): as of the visualisation-unification pipeline
# change, connected-components clustering is now ALSO run for simulated
# data (upstream, by cluster_distance_file.py -- outside this script), so
# it can be compared against the ground truth the same way every other
# method is. Nothing below needs to special-case which pipeline produced
# it: get_dfs_from_sketch (simulation path) already groups purely by
# whatever "method" values are present in clusters.tsv, and every
# downstream lookup keys off CONNECTED_COMPONENTS_METHOD_NAMES /
# SKETCH_METHOD_NAMES / COMBO_ORDER / FANCYDICT, all of which already list
# these methods regardless of pipeline. Simulated clusters.tsv files that
# don't (yet) contain connected-components rows are unaffected: the method
# simply won't appear in that assembly/seed's plots, exactly like any
# other clusterer whose output folder is missing/incomplete.

CONNECTED_COMPONENTS_THRESHOLDS = (0.6, 0.7)
CONNECTED_COMPONENTS_METHOD_NAMES = [
    f"connected_components_t{threshold}" for threshold in CONNECTED_COMPONENTS_THRESHOLDS
]

# SKETCH_METHOD_NAMES is the union of every sub-method name that can appear
# under the "sketch" clusterer's clusters.tsv "method" column, across BOTH
# pipelines. Historically, simulated-data runs only ever emitted
# SIM_ONLY_SKETCH_METHOD_NAMES + "hdbscan_umap" while real-data runs only
# ever emitted CONNECTED_COMPONENTS_METHOD_NAMES + "hdbscan_umap" (see
# cluster_distance_file.py's --sparse branch); simulated data can now also
# emit CONNECTED_COMPONENTS_METHOD_NAMES (see note above). Keeping one
# combined list means every place
# below that families/italicises/excludes-from-c-sweep "sketch" sub-methods
# (by checking membership in SKETCH_METHOD_NAMES) works unchanged for
# whichever pipeline actually produced the data, with no per-mode branching
# needed at each call site.

SKETCH_METHOD_NAMES = (
    SIM_ONLY_SKETCH_METHOD_NAMES + ["hdbscan_umap"] + CONNECTED_COMPONENTS_METHOD_NAMES
)

EMBED_METHOD_NAMES = ["embed_hdbscan_raw", "embed_hdbscan_tsne", "embed_hdbscan_umap"]

# Every "family" of methods that gets its own bracket + footnote in the
# per-clusterer bar/violin/heatmap plots (sketch and embeddings today).
FAMILY_METHOD_NAMES = {
    "Sketching methods": SKETCH_METHOD_NAMES,
    "Embeddings methods": EMBED_METHOD_NAMES,
}

DEFAULT_FIGSIZE = (14, 6.5)

FANCYDICT = {
    "cdhit/nt": "CD-HIT (NT)",
    "mmseqs2/nt": "MMseqs2 (NT)",
    "cdhit/aa": "CD-HIT (AA)",
    "mmseqs2/aa": "MMseqs2 (AA)",
    "diamond/aa" : "Diamond (AA)",
    "panaroo/aa": "Panaroo",
    "ppanggolin/aa" : "Ppanggolin",
    "panta/aa" : "Panta",
    "panx/aa" : "PanX",
    
    "hdbscan_dist/aa": "Sketch - dist* (AA)",
    "hdbscan_tsne/aa": "Sketch - t-SNE* (AA)",
    "hdbscan_umap/aa": "Sketch - UMAP* (AA)",

    "hdbscan_dist/nt": "Sketch - dist* (NT)",
    "hdbscan_tsne/nt": "Sketch - t-SNE* (NT)",
    "hdbscan_umap/nt": "Sketch - UMAP* (NT)",

    # connected-components sweep,
    # one entry per threshold, per seqtype ===
    **{
        f"connected_components_t{threshold}/aa": f"Connected components t={threshold}* (AA)"
        for threshold in CONNECTED_COMPONENTS_THRESHOLDS
    },
    **{
        f"connected_components_t{threshold}/nt": f"Connected components t={threshold}* (NT)"
        for threshold in CONNECTED_COMPONENTS_THRESHOLDS
    },

    "embed_hdbscan_raw/aa": "Embeddings - dist* (AA)",
    "embed_hdbscan_tsne/aa": "Embeddings - t-SNE* (AA)",
    "embed_hdbscan_umap/aa": "Embeddings - UMAP* (AA)",
}

COMBO_ORDER = [
    "cdhit/aa",
    "cdhit/nt",
    "mmseqs2/aa",
    "mmseqs2/nt",
    "diamond/aa",
    "panaroo/aa",
    "ppanggolin/aa",
    "panta/aa",
    "panx/aa",
    "hdbscan_dist/aa",
    "hdbscan_tsne/aa",
    "hdbscan_umap/aa",
    "hdbscan_dist/nt",
    "hdbscan_tsne/nt",
    "hdbscan_umap/nt",
    *[f"connected_components_t{threshold}/aa" for threshold in CONNECTED_COMPONENTS_THRESHOLDS],
    *[f"connected_components_t{threshold}/nt" for threshold in CONNECTED_COMPONENTS_THRESHOLDS],
    "embed_hdbscan_raw/aa",
    "embed_hdbscan_tsne/aa",
    "embed_hdbscan_umap/aa",
]

SKETCH_FOOTNOTE = (
    "* Sketch sub-methods (HDBSCAN on distance/t-SNE/UMAP, HDBSCAN on UMAP, "
    "and a single-linkage connected-components threshold sweep) run once "
    "per seed on a fixed sketch distance matrix; there is no c (minimum "
    "sequence identity) sweep, so no averaging over c is performed."
)

EMBED_FOOTNOTE = (
    "* Embeddings/HDBSCAN methods run once per seed on a fixed embedding; "
    "there is no c (minimum sequence identity) sweep, so no averaging over c is performed."
)

FAMILY_FOOTNOTES = {
    "Sketching methods": SKETCH_FOOTNOTE,
    "Embeddings methods": EMBED_FOOTNOTE,
}

CONFIGDICT = {
    "adj_rand_index": {
        "ylabel": "Adjusted Rand index (adim.)",
        "ylimits": (0.1, 1.1),
        "ylimits_c": (0.4, 1.1),
    },
    "purity": {"ylabel": "Purity (adim.)", "ylimits": (0.1, 1.1), "ylimits_c": (0.4, 1.1)},
    "adj_mutual_info": {
        "ylabel": "Adjusted mutual information (adim.)",
        "ylimits": (0.1, 1.1),
        "ylimits_c": (0.4, 1.1),
    },
    "v_measure": {"ylabel": "V-measure (adim.)", "ylimits": (0.1, 1.1)},
    "runtime": {"ylabel": "Runtime (s)"},
}

CONFIGDICT_COLOURS = {
    "cdhit/aa":        "#FF6B9A",
    "cdhit/nt":        "#C73E72",

    "mmseqs2/aa":      "#A78BFA",
    "mmseqs2/nt":      "#7C5CE0",

    "diamond/aa":      "#67B7F7",

    "panaroo/aa":      "#FF8A5B",

    "ppanggolin/aa":   "#FFCF5A",

    "panta/aa":        "#4ECDC4",

    "panx/aa":         "#B39DDB",

    "hdbscan_dist/aa": "#7BC043",
    "hdbscan_tsne/aa": "#4C9A2A",
    "hdbscan_umap/aa": "#A3D977",

    "hdbscan_dist/nt": "#5DA12F",
    "hdbscan_tsne/nt": "#34751B",
    "hdbscan_umap/nt": "#7FBE5A",

    **{
        f"connected_components_t{threshold}/aa": colour
        for threshold, colour in zip(
            CONNECTED_COMPONENTS_THRESHOLDS,
            ["#52C285", "#7ADBA0"],
        )
    },
    **{
        f"connected_components_t{threshold}/nt": colour
        for threshold, colour in zip(
            CONNECTED_COMPONENTS_THRESHOLDS,
            ["#358C64", "#4FA47C"],
        )
    },

    "embed_hdbscan_raw/aa": "#F2A03D",
    "embed_hdbscan_tsne/aa": "#D9770B",
    "embed_hdbscan_umap/aa": "#F7C177",
}

# --- method-group colour coding for the NJ tree plots -----------------
# Three high-level groups of methods, requested for consistent colour
# coding on the NJ tree tip labels (one colour per group, applied
# consistently across every NJ tree plot):
#   1. Sequence clustering / similarity methods -- CD-HIT, MMseqs2,
#      Diamond (AA).
#   2. Pangenome methods -- Panaroo, Ppanggolin, Panta, PanX.
#   3. Sketching / embedding methods -- Sketch (all sub-methods:
#      hdbscan_dist/hdbscan_tsne/hdbscan_umap and the connected-components
#      sweep) and Embeddings (all embed_* sub-methods).
# Keyed by the "clusterer" prefix of a combo key (combo.split("/")[0]),
# so it works directly off the same `x` list every heatmap/tree already
# carries.
METHOD_GROUP_NAMES = {
    "Sequence clustering / similarity": ["cdhit", "mmseqs2", "diamond"],
    "Pangenome methods": ["panaroo", "ppanggolin", "panta", "panx"],
    "Sketching / embedding methods": (
        ["sketch", "embeddings"] + SKETCH_METHOD_NAMES + EMBED_METHOD_NAMES
    ),
}

METHOD_GROUP_COLOURS = {
    "Sequence clustering / similarity": "#3B7DD8",  # blue
    "Pangenome methods": "#2E9E5B",                  # green
    "Sketching / embedding methods": "#D98A2B",      # orange
}

# reverse lookup: clusterer prefix -> group name, built once from the
# dict above so callers don't need to loop over METHOD_GROUP_NAMES.
_METHOD_PREFIX_TO_GROUP = {
    prefix: group
    for group, prefixes in METHOD_GROUP_NAMES.items()
    for prefix in prefixes
}


def get_method_group(combo_key):
    print(f"[TRACE] >>> Entering get_method_group() - defined near line 320 of {__file__}")
    """Map a combo key (e.g. "cdhit/aa", "hdbscan_umap/nt",
    "connected_components_t0.6/aa") to its high-level method group name
    (one of METHOD_GROUP_NAMES's keys), for consistent colour coding.
    Returns None if the clusterer prefix isn't recognised."""
    prefix = combo_key.split("/")[0]
    return _METHOD_PREFIX_TO_GROUP.get(prefix)


# --- shared title helper for real-data mode -----------------------------
# In "real_data" mode every figure's title ends with
# "... -- {namedict[assembly]} ({datatype})", but there is only ever one
# real-data "assembly" (namedict[assembly] is always the literal string
# "Real data" and datatype is always the literal string "real_data"), so
# appending "(real_data)" to the title is pure redundancy -- the title
# already says "Real data" via namedict[assembly]. We therefore never add
# the "(datatype)" suffix in real_data mode, on any figure.
# "simulations" mode is unaffected (namedict[assembly] differs per
# assembly there, so the suffix is never redundant) -- it always keeps
# its "(datatype)" suffix.


def get_datatype_title_suffix(datatype):
    print(f"[TRACE] >>> Entering get_datatype_title_suffix() - defined near line 320 of {__file__}")
    """Return " ({datatype})" the way every title used to hard-code it,
    except that for "real_data" this is never returned, since
    namedict[assembly] is always the literal string "Real data" already,
    so appending "(real_data)" would just duplicate it in every title.
    Other datatypes (e.g. "simulations") always get the suffix, since
    it's not redundant there."""
    if datatype != "real_data":
        return f" ({datatype})"
    return ""


def write_metric_csv(df, outfolder, filename_stub):
    print(f"[TRACE] >>> Entering write_metric_csv() - defined near line 320 of {__file__}")
    """Shared CSV-writer for every numeric metric exported alongside the
    plots (requirement 6): writes `df` (already containing full, unrounded
    precision) to "<outfolder>/csv/<filename_stub>.csv", creating the csv/
    subdirectory if needed. Mirrors save_figure's per-format subdirectory
    convention (png/, pdf/, svg/) with a matching csv/ subdirectory, so
    every numeric export sits alongside its figure counterpart and is easy
    to find from the filename_stub alone.

    Input:
        df             -- a pandas DataFrame with the exact values used to
                           draw the corresponding plot (no rounding).
        outfolder      -- the run's top-level output folder.
        filename_stub  -- file name without extension, ideally matching
                           (or clearly related to) the plot's own stub.
    Output: none (writes "<outfolder>/csv/<filename_stub>.csv").
    """
    csv_dir = os.path.join(outfolder, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(os.path.join(csv_dir, f"{filename_stub}.csv"), index=False)


def jensen_shannon_divergence(p, q, base=2.0):
    print(f"[TRACE] >>> Entering jensen_shannon_divergence() - defined near line 320 of {__file__}")
    """Jensen-Shannon divergence between two discrete distributions `p`
    and `q` (same-length, non-negative arrays; need not already be
    normalised -- they're renormalised to sum to 1 here). Symmetric and
    bounded in [0, 1] when `base=2.0` (the default), since JSD(P||Q) =
    0.5*KL(P||M) + 0.5*KL(Q||M) with M = 0.5*(P+Q), and log base 2 makes
    the maximum divergence (no shared support) equal to 1 bit.

    Zero handling: KL terms only sum over indices where the numerator
    probability is > 0 (the standard 0*log(0/x) := 0 convention), and
    since M = 0.5*(P+Q) is always > 0 wherever P (or Q) is > 0, this never
    divides by zero. Categories where both P and Q are 0 simply drop out
    of the sum, which is the mathematically correct contribution (0).

    Input:  p, q -- array-likes of non-negative numbers, same length.
            base -- logarithm base; 2.0 gives divergence in bits
                    (bounded [0, 1]), None gives natural-log nats.
    Output: float, the Jensen-Shannon divergence between p and q.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p_sum = p.sum()
    q_sum = q.sum()
    p = p / p_sum if p_sum > 0 else p
    q = q / q_sum if q_sum > 0 else q
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    if base is not None:
        jsd = jsd / np.log(base)
    # guard against tiny negative values from floating-point noise
    return max(0.0, float(jsd))


def nicesp(uglysp):
    print(f"[TRACE] >>> Entering nicesp() - defined at line 290 of {__file__}")
    """Turn a dotted species/assembly identifier (e.g. "genus.species") into
    a human-readable, title-cased string for use in plot titles/labels.

    Input:  uglysp -- a string with words separated by "." (dots).
    Output: the same words joined by spaces, with the whole string
            capitalised (first letter upper-case, rest unchanged by
            str.capitalize's usual rules).
    """
    return " ".join(uglysp.split(".")).capitalize()


def get_font_properties(args):
    print(f"[TRACE] >>> Entering get_font_properties() - defined at line 302 of {__file__}")
    """Build the three matplotlib FontProperties objects (regular, italic,
    bold) used throughout the plotting functions in this script, from the
    font file paths supplied on the command line.

    Input:  args -- parsed argparse namespace with .font_regular,
            .font_italic and .font_bold attributes (paths to .ttf files).
    Output: a 3-tuple (regular, italic, bold) of FontProperties objects,
            unpacked by every plotting function as
            `ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props`.
    """
    return (
        FontProperties(fname=args.font_regular),
        FontProperties(fname=args.font_italic),
        FontProperties(fname=args.font_bold),
    )


def get_param_dict_from_splits(thesplits):
    print(f"[TRACE] >>> Entering get_param_dict_from_splits() - defined at line 320 of {__file__}")
    """Parse a list of "key-value" tokens (as produced by splitting a
    result-folder name like "mmseqs2_st-aa_c-0.9" on "_") into a
    dictionary of clustering parameters.

    Input:  thesplits -- list of strings, each expected to be of the form
            "<key>-<value>" (e.g. "st-aa", "c-0.9").
    Output: dict mapping key -> value, where value is cast to float if it
            looks numeric (digits/decimal point only), otherwise kept as
            a string (e.g. {"st": "aa"} or {"c": 0.9}).
    Raises: ValueError if a token doesn't split into exactly two parts on
            "-", so malformed folder names fail loudly instead of
            silently producing a wrong parameter set.
    """
    outdict = {}
    for split in thesplits:
        subsplits = split.split("-")
        if len(subsplits) != 2:
            raise ValueError(f"Malformed parameter token {split!r}; expected key-value")
        if subsplits[1].replace(".", "").isdigit():
            outdict[subsplits[0]] = float(subsplits[1])
        else:
            outdict[subsplits[0]] = subsplits[1]
    return outdict


def get_labels_list_from_df(indf):
    print(f"[TRACE] >>> Entering get_labels_list_from_df() - defined at line 346 of {__file__}")
    """Convert a dense cluster-membership matrix into parallel gene/label
    lists, i.e. flatten "which cluster is each gene in" out of the matrix
    representation used elsewhere in this script.

    Input:  indf -- a (cluster_id x gene_id) DataFrame where a cell is
            >= 0 (e.g. a CD-HIT identity score, or 1.0 for a hard
            membership) if that gene belongs to that cluster, and -1.0
            (or similar sentinel) if it does not.
    Output: (genes, labels) -- two same-length lists; genes[i] is a gene
            id (column) and labels[i] is the cluster_id (index value) it
            was assigned to. Genes with no cluster assignment at all
            (e.g. dropped/filtered by the clusterer, left as -1 everywhere)
            are simply skipped rather than forced into a fake cluster,
            since a clusterer choosing to discard a gene is different
            from it truly belonging to a cluster.
    """
    genes = []
    labels = []
    for column in indf.columns:
        tmp = indf[indf[column] >= 0.0].index
        #if len(tmp) > 1:
            #print(tmp)
        if len(tmp) == 0:
            continue
        genes.append(column)
        labels.append(tmp[0])
    return genes, labels


def get_purity(inlab, truthdf, gene_ids=None):
    print(f"[TRACE] >>> Entering get_purity() - defined at line 376 of {__file__}")
    """Compute clustering purity of a predicted clustering against a
    ground-truth clustering: for each TRUE class, count how many of its
    genes fall into each PREDICTED cluster, keep only the best-matching
    predicted cluster's count, and sum these "best matches" over all true
    classes, normalised by the total number of genes.

    Purity ranges from 0 to 1 (higher = better agreement with ground
    truth); a purity of 1 means every true class maps entirely into a
    single predicted cluster (no cross-contamination), while a low purity
    means genes belonging to the same true class are scattered across
    many different predicted clusters.

    Input:
        inlab   -- list/array of predicted cluster labels, aligned with
                   gene_ids (or with truthdf.index if gene_ids is None).
        truthdf -- ground-truth membership DataFrame: one column per true
                   class, one row per gene, boolean True where that gene
                   belongs to that true class.
        gene_ids -- optional list of gene ids that inlab's entries refer
                   to (needed because clusterers can drop genes, so
                   `inlab` is not guaranteed to be aligned to
                   truthdf.index by position any more -- see the note
                   below).
    Output: float purity score in [0, 1].
    """
    # Recode les labels to obtain consecutive numbers
    le = LabelEncoder()
    inlab_encoded = le.fit_transform(inlab)
    nclusters = len(le.classes_)

    # gene_ids gives the gene id that each entry of inlab/inlab_encoded refers
    # to. Previously this function assumed inlab was positionally aligned to
    # gene number (position i == geneid_i), which only held because every
    # clusterer's matrix was padded to the full contiguous 0..N gene range.
    # Now that we only keep genes actually present in both tables (see
    # calculate_values_from_cluster_matrix), we look genes up explicitly by id.
    if gene_ids is None:
        gene_ids = list(truthdf.index)
    gene_to_encoded = dict(zip(gene_ids, inlab_encoded))

    sumofmaxes = 0

    # For each true class (column of truthdf), tally how many of its
    # member genes ended up in each predicted cluster (countlist), then
    # keep only the largest tally -- i.e. assume every gene in this true
    # class "should" have landed in whichever predicted cluster captured
    # the most of them, and count that as correctly classified.
    for column in truthdf.columns:
        countlist = [0] * nclusters
        for gene in truthdf[truthdf[column] == True].index:
            if gene not in gene_to_encoded:
                continue
            cluster_label = gene_to_encoded[gene]
            countlist[cluster_label] += 1
        sumofmaxes += max(countlist)
    return float(sumofmaxes) / float(len(truthdf.index))

# multiple statistics : purity from the above function & metrics from skicit learn
# The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples 
# and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings
# The Mutual Information is a measure of the similarity between two labels of the same data.
# for mathematical expression see documentation

def calculate_values_from_cluster_matrix(infotuple, indf, truthlab, truthdf):
    print(f"[TRACE] >>> Entering calculate_values_from_cluster_matrix() - defined at line 440 of {__file__}")
    """Compare one clusterer's predicted clustering against the simulation's
    known ground-truth clustering, and compute a full row of
    clustering-agreement statistics (simulation pipeline only -- real data
    has no ground truth and never calls this function).

    Input:
        infotuple -- tuple of bookkeeping values (e.g. assembly, seed,
                     clusterer name) copied straight through into the
                     output row.
        indf      -- this clusterer's (cluster_id x gene_id) membership
                     matrix, as returned by get_df_from_clusterer.
        truthlab  -- list of ground-truth cluster labels, aligned to
                     truthdf.index.
        truthdf   -- ground-truth membership DataFrame (see get_purity).
    Output: a flat list (one row) combining the bookkeeping fields with:
        adjusted Rand index, purity, adjusted mutual information, and
        homogeneity/completeness/V-measure -- the same set of metrics
        recorded in `outdf` and later plotted as boxplots/pointplots/
        heatmaps for the simulation pipeline. All of these range roughly
        0 (no agreement, worse than random for AMI/ARI) to 1 (perfect
        agreement with the ground truth), so higher is always better.

        ARI and AMI are also each accompanied by an empirical permutation
        p-value (adj_rand_index_pvalue, adj_mutual_info_pvalue -- see
        permutation_test_agreement) testing whether that method's
        agreement with the ground truth is stronger than chance label
        assignment would produce. Purity/homogeneity/completeness/
        V-measure are not accompanied by a p-value: unlike ARI (chance-
        corrected pairwise agreement) and AMI (chance-corrected mutual
        information), those metrics don't have the same "expected value
        under random labelling" framing that a label-shuffling null
        directly tests, so no p-value is computed for them here.
    """
    genes_present, probelab = get_labels_list_from_df(indf)

    # ---------------------------------------------------------------
    # Shape matching between predicted clustering and ground truth
    # ---------------------------------------------------------------
    # Some clusterers (panaroo, ppanggolin, panta, ...) drop/filter some
    # genes, so `genes_present` can end up smaller than truthdf's genes.
    # Instead of lumping all the missing genes into one artificial "bin"
    # cluster to force matching sizes (see the now-commented-out code in
    # get_df_from_clusterer), we instead take a deepcopy of the ground
    # truth and restrict it down to exactly the genes this clusterer
    # actually assigned to a cluster, so the comparison is like-for-like.
    if set(genes_present) != set(truthdf.index):
        gene_to_truthlabel = dict(zip(truthdf.index, truthlab))

        # keep only genes that exist in both tables, in probelab's order
        matched_genes = [g for g in genes_present if g in gene_to_truthlabel]

        gene_to_probelabel = dict(zip(genes_present, probelab))
        probelab_matched = [gene_to_probelabel[g] for g in matched_genes]
        truthlab_matched = [gene_to_truthlabel[g] for g in matched_genes]
        truthdf_matched = copy.deepcopy(truthdf).loc[matched_genes]
    else:
        matched_genes = genes_present
        probelab_matched = probelab
        truthlab_matched = truthlab
        truthdf_matched = truthdf

    # Empirical permutation-test p-values for ARI and AMI against the
    # ground truth (see permutation_test_agreement's docstring for the
    # method and its rationale). Each call reuses the already-matched
    # label lists above, so it is testing exactly the same comparison as
    # the point estimate computed just below it. Seeded deterministically
    # from (assembly, seed, clusterer/method, metric) so re-running the
    # pipeline on the same inputs reproduces the same p-values.
    ari_seed = _stable_permutation_seed(infotuple, "ari")
    ami_seed = _stable_permutation_seed(infotuple, "ami")
    _, ari_pvalue = permutation_test_agreement(
        truthlab_matched, probelab_matched,
        metric_function=metrics.adjusted_rand_score,
        nperm=AGREEMENT_PVALUE_NPERM, seed=ari_seed,
    )
    _, ami_pvalue = permutation_test_agreement(
        truthlab_matched, probelab_matched,
        metric_function=adjusted_mutual_info_score,
        nperm=AGREEMENT_PVALUE_NPERM, seed=ami_seed,
    )

    outlist = [
        True,
        infotuple[0],
        infotuple[1],
        infotuple[2],
        # Adjusted Rand index: pairwise agreement between the predicted and
        # true clusterings, corrected for chance grouping; 0 = random, 1 = perfect.
        float(metrics.adjusted_rand_score(truthlab_matched, probelab_matched)),
        # Empirical permutation p-value for the ARI above (see
        # permutation_test_agreement): fraction of label-shuffled ARI
        # scores >= the observed ARI. Small p-value => this clusterer's
        # agreement with the ground truth is unlikely to be chance.
        ari_pvalue,
        # Purity: fraction of genes whose predicted cluster matches the
        # majority true class of that predicted cluster (see get_purity).
        get_purity(probelab_matched, truthdf_matched, matched_genes),
        # Adjusted mutual information: information-theoretic agreement
        # score, corrected for chance, between predicted and true labels.
        float(adjusted_mutual_info_score(truthlab_matched, probelab_matched)),
        # Empirical permutation p-value for the AMI above, same method as
        # ari_pvalue but using AMI as the test statistic.
        ami_pvalue,
    ]
    # Homogeneity (each predicted cluster contains only members of a
    # single true class), completeness (all members of a true class are
    # assigned to the same predicted cluster), and V-measure (their
    # harmonic mean) -- standard sklearn cluster-agreement metrics.
    outlist += [float(el) for el in metrics.homogeneity_completeness_v_measure(truthlab_matched, probelab_matched)]
    return outlist


def _stable_permutation_seed(infotuple, suffix):
    print(f"[TRACE] >>> Entering _stable_permutation_seed() - defined near line 660 of {__file__}")
    """Deterministic RNG seed for permutation_test_agreement, derived from
    the (assembly, seed, clusterer) row identity plus which metric it's
    for (`suffix`, e.g. "ari"/"ami"). Using a fixed seed per row+metric
    (instead of leaving `seed=None`, which would draw a fresh, unlogged
    seed from OS entropy every run) means re-running the pipeline on
    identical inputs reproduces identical p-values, while still using an
    independent permutation stream per metric/row (so ARI and AMI don't
    share a null distribution by accident).

    Input:  infotuple -- the same (assembly, seed, clusterer) tuple passed
                into calculate_values_from_cluster_matrix.
            suffix     -- short string identifying which metric this seed
                is for (e.g. "ari", "ami").
    Output: a 32-bit unsigned int suitable for numpy.random.default_rng.
    """
    key = "_".join(str(el) for el in infotuple) + f"_{suffix}"
    return zlib.crc32(key.encode("utf-8"))


def permutation_test_agreement(labels1, labels2, metric_function=metrics.adjusted_rand_score, nperm=10000, seed=None):
    print(f"[TRACE] >>> Entering permutation_test_agreement() - defined at line 544 of {__file__}")
    """Empirical permutation test for whether the agreement between two
    label sets (e.g. predicted vs. true clustering) is stronger than
    would be expected by chance.

    Algorithm: compute the observed agreement score, then repeatedly
    shuffle `labels1` at random (breaking any real correspondence to
    `labels2`) and recompute the score each time; the p-value is the
    fraction of these random-shuffle scores that are >= the observed
    score (i.e. how often chance alone could produce as strong or
    stronger an agreement). A "+1" is added to both numerator and
    denominator (add-one/Laplace correction) so the p-value can never be
    reported as exactly 0, which would be misleading with a finite number
    of permutations.

    Input:
        labels1, labels2 -- two same-length label sequences to compare.
        metric_function   -- agreement metric to use (default: Adjusted
                              Rand index).
        nperm             -- number of random permutations to draw.
        seed              -- optional RNG seed for reproducibility.
    Output: (observed_score, pvalue) -- the true (unshuffled) metric
        value, and its empirical permutation p-value. A small p-value
        means the observed agreement is unlikely to have arisen from
        random label assignment, i.e. the clustering genuinely agrees
        with the reference more than chance would predict.
    """
    rng = default_rng(seed)
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)

    observed = metric_function(labels1, labels2)

    # Null distribution: shuffle labels1 nperm times and recompute the
    # metric against the (unshuffled) labels2 each time, to see how large
    # an agreement score chance alone can produce.
    permuted = np.empty(nperm)
    for i in range(nperm):
        permuted[i] = metric_function(rng.permutation(labels1), labels2)

    n_greater = np.sum(permuted >= observed)
    pvalue = float((n_greater + 1) / (nperm + 1))
    return observed, pvalue


def parse_cdhit_identity(line):
    print(f"[TRACE] >>> Entering parse_cdhit_identity() - defined at line 589 of {__file__}")
    """Extract the sequence-identity percentage from one member line of a
    CD-HIT .clstr file (e.g. "...at 95.00%" or "...at 1:98:99/95%") and
    convert it to a fraction in [0, 1].

    Input:  line -- a single line of text from a .clstr file, member entry.
    Output: float sequence identity, e.g. 0.95 for "95.00%".
    """
    identity = line.strip().split("at", 1)[1].strip().replace("%", "")
    if "/" in identity:
        identity = identity.split("/")[-1]
    return float(identity) / 100.0


def get_df_from_clusterer(clusterer, folderpath, true_max_gene=None):
    print(f"[TRACE] >>> Entering get_df_from_clusterer() - defined at line 603 of {__file__}")
    """Parse one clustering tool's raw output files (simulation pipeline)
    into a common dense (cluster_id x gene_id) membership matrix, so every
    downstream function can treat all clusterers uniformly regardless of
    their native output format.

    Input:
        clusterer     -- name of the tool whose output to parse; one of
                          the entries in CLUSTERERS (cdhit, mmseqs2,
                          diamond, panaroo, ppanggolin, panta, panx,
                          sketch, embeddings).
        folderpath    -- path to this tool's result folder for one
                          assembly/seed/parameter combination.
        true_max_gene -- highest simulated gene index expected (genes are
                          named "geneid_0".."geneid_true_max_gene" in the
                          simulation), used to pad the matrix so every
                          clusterer's output covers the same full gene
                          universe, even genes it silently dropped.
    Output: a pandas DataFrame indexed by cluster_id, one column per gene
        id, where a cell is >= 0 (membership, e.g. a CD-HIT sequence
        identity score, or 1.0) if that gene belongs to that cluster, and
        -1.0 if it does not. Each clusterer's native output format (CD-HIT
        .clstr, MMseqs2/DIAMOND 2-column TSV, Panaroo's internal CD-HIT
        clustering + gene_data.csv id lookup, ppanggolin's TSV, panta's
        TSV, panX's JSON orthogroups, or the sketch/embeddings HDBSCAN
        label files) is parsed by its own branch below into this same
        common shape.
    """
    if clusterer == "cdhit":
        listoflists = []
        setofgenes = set()
        listofclusters = []
        tmpdict = {}
        with open(os.path.join(folderpath, "cdhit.clstr"), "r") as f:
            tmpclusterid = -1
            for line in f:
                if line[0] == ">":
                    tmpclusterid = int(line.replace(">", "").split(" ")[1].strip())
                    tmpdict[tmpclusterid] = {}
                    listofclusters.append(tmpclusterid)
                else:
                    tmpgeneid = line.strip().split(">")[1].split("...")[0]
                    setofgenes.add(tmpgeneid)
                    tmpdict[tmpclusterid][tmpgeneid] = (
                        parse_cdhit_identity(line)
                    ) if "*" not in line else 2.0

        listofgenes = list(setofgenes)
        listofgenes.sort(key=lambda x: int(x.split("_")[1]))
        for cluster in listofclusters:
            row = [cluster]
            for gene in listofgenes:
                row.append(tmpdict[cluster][gene] if gene in tmpdict[cluster] else -1.0)
            listoflists.append(row)
        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + listofgenes)
        return outdf.set_index("cluster_id")

    if clusterer == "mmseqs2":
        firstdf = pd.read_csv(
            os.path.join(folderpath, "mmseqs2_cluster.tsv"),
            names=["cluster_id", "gene_id"],
            sep="\t",
        )
        clusterlist = list(set([int(iC.split("_")[1]) for iC in firstdf["cluster_id"]]))
        clusterlist.sort()
        genelist = list(set(list(firstdf["gene_id"])))
        genelist.sort(key=lambda x: int(x.split("_")[1]))

        listoflists = []
        for cluster_index in range(len(clusterlist)):
            tmpset = set(
                firstdf[
                    firstdf["cluster_id"] == "geneid_" + str(clusterlist[cluster_index])
                ]["gene_id"]
            )
            row = [cluster_index]
            for gene in genelist:
                row.append(+1.0 if gene in tmpset else -1.0)
            listoflists.append(row)

        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + genelist)

        return outdf.set_index("cluster_id")
    
    if clusterer == "diamond":
        # read clusters
        # the output for diamond is a 2-column file (ascii text)
        # then copy paste from mmseqs2 lecture code (identical output => same code)

        firstdf = pd.read_csv(
            os.path.join(folderpath, "diamond"),
            names=["cluster_id", "gene_id"],
            sep="\t",
        )
        clusterlist = list(set([int(iC.split("_")[1]) for iC in firstdf["cluster_id"]]))
        clusterlist.sort()
        genelist = list(set(list(firstdf["gene_id"])))
        genelist.sort(key=lambda x: int(x.split("_")[1]))

        listoflists = []
        for cluster_index in range(len(clusterlist)):
            tmpset = set(
                firstdf[
                    firstdf["cluster_id"] == "geneid_" + str(clusterlist[cluster_index])
                ]["gene_id"]
            )
            row = [cluster_index]
            for gene in genelist:
                row.append(+1.0 if gene in tmpset else -1.0)
            listoflists.append(row)

        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + genelist)

        return outdf.set_index("cluster_id")

    if clusterer == "panaroo":

        gene_data = pd.read_csv(
            os.path.join(folderpath, "panaroo/gene_data.csv"),
            low_memory=False,
            header=None
        )

        panaroo_to_original = dict(
            zip(
                gene_data[2],
                gene_data[3]
            )
        )

        listoflists = []
        setofgenes = set()
        listofclusters = []
        tmpdict = {}

        with open(os.path.join(folderpath, "panaroo/combined_protein_cdhit_out.txt.clstr"), "r") as f:
            tmpclusterid = -1

            for line in f:
                if line[0] == ">":
                    tmpclusterid = int(line.replace(">", "").split(" ")[1].strip())
                    tmpdict[tmpclusterid] = {}
                    listofclusters.append(tmpclusterid)

                else:
                    panaroo_geneid = line.strip().split(">")[1].split("...")[0]

                    tmp = panaroo_to_original.get(
                        panaroo_geneid,
                        panaroo_geneid
                    )

                    match = re.search(r'geneid_\d+(?:_iso_\d+)?', tmp)
                    tmpgeneid = match.group(0) if match else None

                    setofgenes.add(tmpgeneid)
                    tmpdict[tmpclusterid][tmpgeneid] = (
                        parse_cdhit_identity(line)
                        if "*" not in line
                        else 2.0
                    )

        # Get all genes
        listofgenes = list(setofgenes)
        listofgenes.sort(
            key=lambda x: int(re.search(r"geneid_(\d+)", x).group(1))
        )

        # ---------------------------------------------------------
        # Add missing gene IDs before creating rows
        # ---------------------------------------------------------
        gene_nums = {
            int(re.search(r"geneid_(\d+)", g).group(1))
            for g in listofgenes
        }

        max_gene = max(gene_nums)
        if true_max_gene is not None:
            max_gene = max(max_gene, true_max_gene)

        missing = [
            f"geneid_{i}"
            for i in range(max_gene + 1)
            if i not in gene_nums
        ]

        if missing:
            listofgenes.extend(missing)

            listofgenes.sort(
                key=lambda x: int(re.search(r"geneid_(\d+)", x).group(1))
            )

        # ---------------------------------------------------------
        # Create normal clusters
        # ---------------------------------------------------------
        for cluster in listofclusters:
            row = [cluster]

            for gene in listofgenes:
                row.append(
                    tmpdict[cluster].get(gene, -1.0)
                )

            listoflists.append(row)

        # ---------------------------------------------------------
        # Create one new cluster containing all missing genes
        # ---------------------------------------------------------
        # NOTE: no longer done -- genes panaroo drops are now simply left
        # unassigned (all -1 across every row) and excluded from the
        # metrics comparison (see get_labels_list_from_df /
        # calculate_values_from_cluster_matrix), rather than being lumped
        # together into one artificial cluster. Kept here, commented out,
        # for reference.
        # if missing:
        #     new_cluster_id = max(listofclusters) + 1
        #
        #     row = [new_cluster_id]
        #
        #     for gene in listofgenes:
        #         row.append(
        #             1.0 if gene in missing else -1.0
        #         )
        #
        #     listoflists.append(row)

        outdf = pd.DataFrame(
            listoflists,
            columns=["cluster_id"] + listofgenes
        )

        return outdf.set_index("cluster_id")

    if clusterer == "ppanggolin":
        output_dir = os.path.join(folderpath, "ppanggolin_outputs/")
        os.makedirs(output_dir, exist_ok=True)

        try:
            subprocess.run(
                [
                    "ppanggolin",
                    "write_pangenome",
                    "-p",
                    os.path.join(folderpath, "ppanggolin/pangenome.h5"),
                    "-o",
                    output_dir,
                    "--families_tsv",
                    "-f",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "ppanggolin write_pangenome failed "
                f"(exit code {exc.returncode}):\n{exc.stderr}"
            ) from exc

        families_file = os.path.join(output_dir, "gene_families.tsv")

        if not os.path.isfile(families_file):
            raise RuntimeError(
                f"PPanGGOLiN did not create expected file: {families_file}"
            )

        # gene_families.tsv has no header, 4 columns:
        # family_id, gene_id (embedded geneid_N[_iso_M]), an always-empty column, fragment flag ("F"/"")
        families = pd.read_csv(
            families_file,
            sep="\t",
            header=None,
            names=["family_id", "gene_id", "_unused", "fragment_flag"],
            low_memory=False,
        )

        def extract_geneid(raw):
            # PPanGGOLiN's gene_id field embeds our simulation's own
            # "geneid_N" (optionally with an "_iso_M" isoform suffix)
            # inside a longer, tool-specific identifier string; pull just
            # that recognisable token back out via regex so downstream
            # code can match genes across tools by a common id.
            match = re.search(r"geneid_\d+(?:_iso_\d+)?", raw)
            return match.group(0) if match else None

        families["gene_id_clean"] = families["gene_id"].apply(extract_geneid)

        if families["gene_id_clean"].isna().any():
            missing_raw = families.loc[families["gene_id_clean"].isna(), "gene_id"].unique()
            raise RuntimeError(
                "Could not parse a geneid_ token out of some ppanggolin gene ids, "
                f"e.g.: {list(missing_raw[:5])}"
            )

        genelist = sorted(
            families["gene_id_clean"].unique(),
            key=lambda x: int(re.search(r"geneid_(\d+)", x).group(1)),
        )

        # ---------------------------------------------------------
        # Add missing gene IDs before creating rows (genes PPanGGOLiN
        # dropped/filtered and that never appear in gene_families.tsv)
        # ---------------------------------------------------------
        gene_nums = {
            int(re.search(r"geneid_(\d+)", g).group(1))
            for g in genelist
        }

        max_gene = max(gene_nums)
        if true_max_gene is not None:
            max_gene = max(max_gene, true_max_gene)

        missing = [
            f"geneid_{i}"
            for i in range(max_gene + 1)
            if i not in gene_nums
        ]

        if missing:
            genelist.extend(missing)
            genelist.sort(
                key=lambda x: int(re.search(r"geneid_(\d+)", x).group(1))
            )

        familylist = sorted(families["family_id"].unique())
        family_to_genes = families.groupby("family_id")["gene_id_clean"].apply(set)

        # ---------------------------------------------------------
        # Create normal clusters
        # ---------------------------------------------------------
        listoflists = []
        for cluster_index, family in enumerate(familylist):
            genes_in_family = family_to_genes[family]
            row = [cluster_index] + [
                1.0 if gene in genes_in_family else -1.0 for gene in genelist
            ]
            listoflists.append(row)

        # ---------------------------------------------------------
        # Create one new cluster containing all missing genes
        # ---------------------------------------------------------
        # NOTE: no longer done -- see equivalent note in the panaroo branch
        # above. Kept here, commented out, for reference.
        # if missing:
        #     new_cluster_id = len(familylist)
        #     missingset = set(missing)
        #     row = [new_cluster_id] + [
        #         1.0 if gene in missingset else -1.0 for gene in genelist
        #     ]
        #     listoflists.append(row)

        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + genelist)

        return outdf.set_index("cluster_id")

    if clusterer == "panta":
        clusters_file = os.path.join(folderpath, "panta/annotated_clusters.json")
        if not os.path.isfile(clusters_file):
            raise RuntimeError(
                f"Panta did not create expected file: {clusters_file}"
            )

        with open(clusters_file, "r") as f:
            clusters = json.load(f)

        def extract_geneid(raw):
            # Same rationale as the ppanggolin branch above: Panta's own
            # gene id string embeds our simulation's "geneid_N" token
            # inside extra tool-specific formatting, so pull it back out
            # via regex to get a common id usable across tools.
            match = re.search(r"geneid_\d+(?:_iso_\d+)?", raw)
            return match.group(0) if match else None

        group_to_genes = {}
        setofgenes = set()
        for group, groupinfo in clusters.items():
            genes_clean = set()
            for raw_gene in groupinfo["gene_id"]:
                geneid_clean = extract_geneid(raw_gene)
                if geneid_clean is None:
                    raise RuntimeError(
                        f"Could not parse a geneid_ token out of panta gene id: {raw_gene!r}"
                    )
                genes_clean.add(geneid_clean)
                setofgenes.add(geneid_clean)
            group_to_genes[group] = genes_clean

        # group keys are "groups_N" and are NOT lexically sortable (groups_10 < groups_2)
        grouplist = sorted(
            clusters.keys(),
            key=lambda g: int(re.search(r"\d+", g).group(0)),
        )

        genelist = sorted(
            setofgenes,
            key=lambda x: int(re.search(r"geneid_(\d+)", x).group(1)),
        )

        # ---------------------------------------------------------
        # Add missing gene IDs before creating rows (genes panta
        # dropped/filtered and that never appear in annotated_clusters.json)
        # ---------------------------------------------------------
        gene_nums = {
            int(re.search(r"geneid_(\d+)", g).group(1))
            for g in genelist
        }

        max_gene = max(gene_nums)
        if true_max_gene is not None:
            max_gene = max(max_gene, true_max_gene)

        missing = [
            f"geneid_{i}"
            for i in range(max_gene + 1)
            if i not in gene_nums
        ]

        if missing:
            genelist.extend(missing)
            genelist.sort(
                key=lambda x: int(re.search(r"geneid_(\d+)", x).group(1))
            )

        # ---------------------------------------------------------
        # Create normal clusters
        # ---------------------------------------------------------
        listoflists = []
        for cluster_index, group in enumerate(grouplist):
            genes_in_group = group_to_genes[group]
            row = [cluster_index] + [
                1.0 if gene in genes_in_group else -1.0 for gene in genelist
            ]
            listoflists.append(row)

        # ---------------------------------------------------------
        # Create one new cluster containing all missing genes
        # ---------------------------------------------------------
        # NOTE: no longer done -- see equivalent note in the panaroo branch
        # above. Kept here, commented out, for reference.
        # if missing:
        #     new_cluster_id = len(grouplist)
        #     missingset = set(missing)
        #     row = [new_cluster_id] + [
        #         1.0 if gene in missingset else -1.0 for gene in genelist
        #     ]
        #     listoflists.append(row)
        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + genelist)
        return outdf.set_index("cluster_id")
    
    if clusterer == "sketch":
        raise RuntimeError(
            "get_df_from_clusterer('sketch', ...) should not be called directly; "
            "use get_dfs_from_sketch(), since one sketch folder can contain "
            "several methods (hdbscan_dist/hdbscan_tsne/hdbscan_umap)."
        )

    if clusterer == "embeddings":
        raise RuntimeError(
            "get_df_from_clusterer('embeddings', ...) should not be called directly; "
            "use get_dfs_from_embeddings(), since one embeddings folder can contain "
            "several methods (hdbscan_dist/hdbscan_tsne/hdbscan_umap)."
        )

    
        return outdf.set_index("cluster_id")

    if clusterer == "panx":
        clusters_file = os.path.join(folderpath, "protein_faa/diamond_matches/allclusters_final.tsv")
        if not os.path.isfile(clusters_file):
            raise RuntimeError(
                f"PanX did not create expected file: {clusters_file}"
            )
 
        def extract_geneid(raw):
            # Same rationale as the ppanggolin/panta branches above: PanX's
            # allclusters_final.tsv embeds the simulation's "geneid_N"
            # token inside its own gene id formatting, so extract it via
            # regex (PanX has no "_iso_M" isoform suffix to worry about,
            # unlike ppanggolin/panta, hence the simpler pattern here).
            match = re.search(r"geneid_\d+", raw)
            return match.group(0) if match else None

        listofclusters = []
        cluster_to_genes = {}
        setofgenes = set()
        n_skipped_lines = 0
 
        with open(clusters_file, "r") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                fields = line.split("\t")
                genes_in_cluster = set()
                corrupted = False
                for field in fields:
                    geneid_clean = extract_geneid(field)
                    if geneid_clean is None:
                        corrupted = True
                        break
                    genes_in_cluster.add(geneid_clean)
 
                if corrupted:
                    n_skipped_lines += 1
                    continue
 
                setofgenes.update(genes_in_cluster)
                cluster_index = len(listofclusters)
                listofclusters.append(cluster_index)
                cluster_to_genes[cluster_index] = genes_in_cluster
        """
        if n_skipped_lines:
            warnings.warn(
                f"panx: skipped {n_skipped_lines} corrupted cluster line(s) "
                f"(missing geneid_ token) in {clusters_file}",
                RuntimeWarning,
                stacklevel=2,
            )
        """
 
        genelist = sorted(
            setofgenes,
            key=lambda x: int(re.search(r"geneid_(\d+)", x).group(1)),
        )
 
        # ---------------------------------------------------------
        # Add missing gene IDs before creating rows (genes panx
        # dropped/filtered and that never appear in allclusters.tsv)
        # ---------------------------------------------------------
        gene_nums = {
            int(re.search(r"geneid_(\d+)", g).group(1))
            for g in genelist
        }
 
        max_gene = max(gene_nums) if gene_nums else -1
        if true_max_gene is not None:
            max_gene = max(max_gene, true_max_gene)
 
        missing = [
            f"geneid_{i}"
            for i in range(max_gene + 1)
            if i not in gene_nums
        ]
 
        if missing:
            genelist.extend(missing)
            genelist.sort(
                key=lambda x: int(re.search(r"geneid_(\d+)", x).group(1))
            )
 
        # ---------------------------------------------------------
        # Create normal clusters
        # ---------------------------------------------------------
        listoflists = []
        for cluster_index in listofclusters:
            genes_in_cluster = cluster_to_genes[cluster_index]
            row = [cluster_index] + [
                1.0 if gene in genes_in_cluster else -1.0 for gene in genelist
            ]
            listoflists.append(row)
 
        # ---------------------------------------------------------
        # Create one new cluster containing all missing genes
        # ---------------------------------------------------------
        # NOTE: no longer done -- see equivalent note in the panaroo branch
        # above. Kept here, commented out, for reference.
        # if missing:
        #     new_cluster_id = len(listofclusters)
        #     missingset = set(missing)
        #     row = [new_cluster_id] + [
        #         1.0 if gene in missingset else -1.0 for gene in genelist
        #     ]
        #     listoflists.append(row)
 
        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + genelist)
    
        return outdf.set_index("cluster_id")


    raise RuntimeError("Clusterer " + clusterer + " not supported!")


# === (real-data support) ===================================
def get_df_from_clusterer_realdata(clusterer, folderpath):
    print(f"[TRACE] >>> Entering get_df_from_clusterer_realdata() - defined at line 1187 of {__file__}")
    """Real-data equivalent of get_df_from_clusterer, restricted to CD-HIT,
    DIAMOND, MMseqs2, and Panaroo (requirement 5).

    Why this needs to be a separate function rather than a tweak to the
    original one: the original cdhit/mmseqs2/diamond/panaroo branches above
    rely on simulation-only assumptions that silently produce wrong results
    (or crash) on real gene IDs:

      1. Gene IDs are assumed to look like "geneid_<N>" so a numeric suffix
         can be parsed out of them (`int(x.split("_")[1])`) purely to sort
         columns. Real gene IDs (locus tags, RefSeq accessions, etc.) have
         no such guaranteed structure, so that parse would raise or silently
         mis-sort. Sort order has no effect on any downstream metric, so we
         just sort lexically here instead.
      2. For mmseqs2/diamond, the original code reconstructs each cluster's
         representative gene ID as the literal string "geneid_" + <int>
         and uses that to re-select rows. This "reconstruction" only works
         because in the simulated data the mmseqs2/diamond cluster_id
         column happens to already be a geneid_N string, and the code was
         extracting the N and rebuilding the same string. On real data,
         cluster_id is still literally the representative gene's own real
         ID (that's the native mmseqs2/diamond output format) -- there is
         nothing to parse or reconstruct, so we group directly on it.
      3. For panaroo, the simulation path recovers the "geneid_N" gene id
         from each Panaroo internal protein id (e.g. "0_0_0") by looking it
         up in gene_data.csv and then regex-extracting a `geneid_\\d+`
         token out of whatever that lookup returns. Real gene IDs (e.g.
         "NLEIDEKG_00001") have no such embedded token to extract, so here
         we instead use the gene_data.csv lookup result directly as the
         gene id -- see the panaroo branch below.

    There is also no `true_max_gene` padding step here: padding existed only
    to align a clusterer's gene universe with the ground-truth gene
    universe (see get_purity/calculate_values_from_cluster_matrix), and
    there is no ground truth for real data to align to. Every gene that
    appears in the clustering output is simply used as-is.
    """
    if clusterer not in REAL_DATA_CLUSTERERS:
        raise ValueError(
            f"get_df_from_clusterer_realdata only supports {REAL_DATA_CLUSTERERS}, "
            f"got {clusterer!r}"
        )

    # NOTE on performance: CD-HIT/MMseqs2/DIAMOND clustering is a *partition*
    # -- each gene belongs to exactly one cluster. Building the dense
    # n_clusters x n_genes matrix (mostly -1.0 padding) with nested Python
    # `for cluster: for gene:` loops is O(n_clusters * n_genes). That was
    # fine on the small simulated benchmark data, but on real data, where
    # gene/cluster counts can run into the tens of thousands, it blows up
    # to billions of iterations in pure Python and looks like the script is
    # stuck/hanging. The fix below builds the exact same DataFrame (same
    # index/columns/values/sort order), but via vectorized pandas
    # construction instead of nested Python loops, which is orders of
    # magnitude faster and scales to real data sizes. This is unrelated to
    # the geneid_N -> string identifier change; that part was already fine.
    if clusterer == "cdhit":
        setofgenes = set()
        listofclusters = []
        tmpdict = {}
        with open(os.path.join(folderpath, "cdhit.clstr"), "r") as f:
            tmpclusterid = -1
            for line in f:
                if line[0] == ">":
                    tmpclusterid = int(line.replace(">", "").split(" ")[1].strip())
                    tmpdict[tmpclusterid] = {}
                    listofclusters.append(tmpclusterid)
                else:
                    tmpgeneid = line.strip().split(">")[1].split("...")[0]
                    setofgenes.add(tmpgeneid)
                    tmpdict[tmpclusterid][tmpgeneid] = (
                        parse_cdhit_identity(line)
                    ) if "*" not in line else 2.0

        # Real gene IDs have no guaranteed numeric structure to sort on;
        # a plain lexical sort gives a deterministic column order and has
        # no bearing on any metric computed downstream.
        listofgenes = sorted(setofgenes)

        # Vectorized equivalent of the old nested-loop dense-matrix build:
        # tmpdict is already {cluster_id: {gene_id: value}}, so
        # pd.DataFrame.from_dict does the row/column alignment in C rather
        # than Python, then we just reindex to the desired column order and
        # fill the "not a member" cells with -1.0.
        outdf = pd.DataFrame.from_dict(tmpdict, orient="index")
        outdf = outdf.reindex(index=listofclusters, columns=listofgenes)
        outdf = outdf.fillna(-1.0)
        outdf.index.name = "cluster_id"
        return outdf

    # mmseqs2 and diamond share an output format: 2 columns, tab separated,
    # (cluster representative gene id, member gene id).
    if clusterer in ("mmseqs2", "diamond"):
        filename = "mmseqs2_cluster.tsv" if clusterer == "mmseqs2" else "diamond"
        firstdf = pd.read_csv(
            os.path.join(folderpath, filename),
            names=["cluster_id", "gene_id"],
            sep="\t",
        )

        # pd.crosstab builds the dense (sorted-cluster x sorted-gene) membership
        # matrix in one vectorized call -- same result as the old
        # "for cluster_index, cluster_id: for gene: ..." double loop, but
        # without the O(n_clusters * n_genes) Python-level iteration. Since
        # clustering is a partition, every (cluster, gene) count is 0 or 1.
        crosstab = pd.crosstab(firstdf["cluster_id"], firstdf["gene_id"])
        outdf = crosstab.astype(float)
        outdf[outdf == 0] = -1.0
        # crosstab's index is the *original* cluster_id, sorted -- the old code
        # discarded that value and just renumbered clusters 0..n-1 in sorted
        # order, so replicate that here for output-compatibility.
        outdf = outdf.reset_index(drop=True)
        outdf.index.name = "cluster_id"
        return outdf

    if clusterer == "panta":
        # Same input file and overall approach as the simulation branch
        # above (get_df_from_clusterer's "panta" branch): parse
        # "panta/annotated_clusters.json" and, for every group, collect
        # the gene ids listed under "gene_id". The only real-data-specific
        # change is how a clean gene id is pulled out of panta's raw gene
        # id string: instead of regex-extracting a "geneid_N" simulation
        # token, real gene ids are taken as the substring after the final
        # "-", e.g. "ERR044869-contig00001-NLEIDEKG_00032" ->
        # "NLEIDEKG_00032" (same rationale as the panaroo/cdhit/mmseqs2/
        # diamond branches above: real gene IDs have no "geneid_N"
        # structure to parse).
        clusters_file = os.path.join(folderpath, "panta/annotated_clusters.json")
        if not os.path.isfile(clusters_file):
            raise RuntimeError(
                f"Panta did not create expected file: {clusters_file}"
            )

        with open(clusters_file, "r") as f:
            clusters = json.load(f)

        def extract_geneid_realdata(raw):
            return raw.rsplit("-", 1)[-1]

        group_to_genes = {}
        setofgenes = set()
        for group, groupinfo in clusters.items():
            genes_clean = set()
            for raw_gene in groupinfo["gene_id"]:
                geneid_clean = extract_geneid_realdata(raw_gene)
                genes_clean.add(geneid_clean)
                setofgenes.add(geneid_clean)
            group_to_genes[group] = genes_clean

        # group keys are "groups_N" and are NOT lexically sortable
        # (groups_10 < groups_2), same as the simulation branch. Real-data
        # panta output isn't guaranteed to use that "groups_N" convention,
        # though, so fall back to a plain lexical sort for any group key
        # without a digit in it instead of crashing.
        def _group_sort_key(g):
            match = re.search(r"\d+", g)
            return (0, int(match.group(0))) if match else (1, g)

        grouplist = sorted(clusters.keys(), key=_group_sort_key)

        # Real gene IDs have no guaranteed numeric structure to sort on;
        # a plain lexical sort gives a deterministic column order and has
        # no bearing on any metric computed downstream (same rationale as
        # the cdhit branch above). No true_max_gene padding either, for
        # the same reason as the rest of this function: there is no
        # ground truth gene universe to align to on real data.
        genelist = sorted(setofgenes)

        listoflists = []
        for cluster_index, group in enumerate(grouplist):
            genes_in_group = group_to_genes[group]
            row = [cluster_index] + [
                1.0 if gene in genes_in_group else -1.0 for gene in genelist
            ]
            listoflists.append(row)

        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + genelist)
        return outdf.set_index("cluster_id")

    if clusterer == "ppanggolin":
        # Same input file and overall approach as the simulation branch
        # above (get_df_from_clusterer's "ppanggolin" branch): run
        # `ppanggolin write_pangenome --families_tsv` against
        # "ppanggolin/pangenome.h5" to materialize "gene_families.tsv",
        # then parse that file. The only real-data-specific change is that
        # PPanGGOLiN's gene_id field is now already the plain real gene id
        # (e.g. "AFKLLJAB_01880") -- there is no embedded "geneid_N"
        # simulation token to regex out, so it's used directly as-is
        # (same rationale as the panta/panx/panaroo/cdhit/mmseqs2/diamond
        # branches above: real gene IDs have no "geneid_N" structure to
        # parse).
        output_dir = os.path.join(folderpath, "ppanggolin_outputs/")
        os.makedirs(output_dir, exist_ok=True)

        try:
            subprocess.run(
                [
                    "ppanggolin",
                    "write_pangenome",
                    "-p",
                    os.path.join(folderpath, "ppanggolin/pangenome.h5"),
                    "-o",
                    output_dir,
                    "--families_tsv",
                    "-f",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "ppanggolin write_pangenome failed "
                f"(exit code {exc.returncode}):\n{exc.stderr}"
            ) from exc

        families_file = os.path.join(output_dir, "gene_families.tsv")

        if not os.path.isfile(families_file):
            raise RuntimeError(
                f"PPanGGOLiN did not create expected file: {families_file}"
            )

        # gene_families.tsv has no header, 4 columns:
        # family_id, gene_id (now the plain real gene id), an always-empty column, fragment flag ("F"/"")
        families = pd.read_csv(
            families_file,
            sep="\t",
            header=None,
            names=["family_id", "gene_id", "_unused", "fragment_flag"],
            low_memory=False,
        )

        # Real gene IDs have no guaranteed numeric structure to sort on;
        # a plain lexical sort gives a deterministic column order and has
        # no bearing on any metric computed downstream (same rationale as
        # the cdhit/panta/panx branches above). No true_max_gene padding
        # either, for the same reason as the rest of this function: there
        # is no ground truth gene universe to align to on real data.
        genelist = sorted(families["gene_id"].unique())

        familylist = sorted(families["family_id"].unique())
        family_to_genes = families.groupby("family_id")["gene_id"].apply(set)

        listoflists = []
        for cluster_index, family in enumerate(familylist):
            genes_in_family = family_to_genes[family]
            row = [cluster_index] + [
                1.0 if gene in genes_in_family else -1.0 for gene in genelist
            ]
            listoflists.append(row)

        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + genelist)
        return outdf.set_index("cluster_id")

    if clusterer == "panx":
        # Same input file and overall approach as the simulation branch
        # above (get_df_from_clusterer's "panx" branch): parse
        # "protein_faa/diamond_matches/allclusters_final.tsv" and, for
        # every line (= one cluster), collect the gene ids listed in its
        # tab-separated fields. The only real-data-specific change is how
        # a clean gene id is pulled out of panX's raw gene id string:
        # instead of regex-extracting a "geneid_N" simulation token, real
        # gene ids are the substring after the "|" delimiter, e.g.
        # "ERR045435|GPPBCAIM_00381" -> "GPPBCAIM_00381" (same rationale
        # as the panta/panaroo/cdhit/mmseqs2/diamond branches above: real
        # gene IDs have no "geneid_N" structure to parse).
        clusters_file = os.path.join(folderpath, "protein_faa/diamond_matches/allclusters_final.tsv")
        if not os.path.isfile(clusters_file):
            raise RuntimeError(
                f"PanX did not create expected file: {clusters_file}"
            )

        def extract_geneid_realdata(raw):
            return raw.rsplit("|", 1)[-1]

        listofclusters = []
        cluster_to_genes = {}
        setofgenes = set()
        n_skipped_lines = 0

        with open(clusters_file, "r") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                fields = line.split("\t")
                genes_in_cluster = set()
                corrupted = False
                for field in fields:
                    if "|" not in field:
                        corrupted = True
                        break
                    geneid_clean = extract_geneid_realdata(field)
                    if not geneid_clean:
                        corrupted = True
                        break
                    genes_in_cluster.add(geneid_clean)

                if corrupted:
                    n_skipped_lines += 1
                    continue

                setofgenes.update(genes_in_cluster)
                cluster_index = len(listofclusters)
                listofclusters.append(cluster_index)
                cluster_to_genes[cluster_index] = genes_in_cluster

        if n_skipped_lines:
            warnings.warn(
                f"panx: skipped {n_skipped_lines} corrupted cluster line(s) "
                f"(missing '|' delimiter) in {clusters_file}",
                RuntimeWarning,
                stacklevel=2,
            )

        # Real gene IDs have no guaranteed numeric structure to sort on;
        # a plain lexical sort gives a deterministic column order and has
        # no bearing on any metric computed downstream (same rationale as
        # the cdhit/panta branches above). No true_max_gene padding either,
        # for the same reason as the rest of this function: there is no
        # ground truth gene universe to align to on real data.
        genelist = sorted(setofgenes)

        listoflists = []
        for cluster_index in listofclusters:
            genes_in_group = cluster_to_genes[cluster_index]
            row = [cluster_index] + [
                1.0 if gene in genes_in_group else -1.0 for gene in genelist
            ]
            listoflists.append(row)

        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + genelist)
        return outdf.set_index("cluster_id")

    # clusterer == "panaroo":
    # gene_presence_absence.csv format (one row per gene cluster):
    #   - columns "Gene", "Non-unique Gene name", "Annotation": metadata,
    #     ignored here.
    #   - every other column is one isolate; the cell value is empty when
    #     that cluster is absent from that isolate ("expected, continue
    #     reading the row"), and otherwise contains the gene id(s) (locus
    #     tags) present in that isolate for this cluster. Panaroo separates
    #     multiple paralogous gene ids in the same cell with ";", so we
    #     split on that (and, defensively, on tabs too).
    # NOTE on "refound" genes: Panaroo inserts additional, Panaroo-recovered
    # gene ids into this table (identifiable by "refound" appearing in the
    # gene id) to patch annotation gaps. Per requirement 2, these must not
    # be used for AMI/ARI (or any other clustering-agreement metric).
    # Rather than dropping them here, they are intentionally KEPT in the
    # returned matrix -- filtering happens one step downstream, in
    # `_process_one_realdata_folder`, right before the gene/label lists are
    # handed to the pairwise AMI/ARI/purity/v-measure/F1 machinery (see
    # `filter_refound_genes` there). Keeping them here also lets the
    # "percentage of added genes" statistic (requirement 3B) be computed
    # directly from this matrix, since it needs to know which genes Panaroo
    # added.
    gpa_path = os.path.join(folderpath, "panaroo/gene_presence_absence.csv")
    gene_presence_absence = pd.read_csv(gpa_path, low_memory=False)

    meta_cols = ["Gene", "Non-unique Gene name", "Annotation"]
    missing_meta = [c for c in meta_cols if c not in gene_presence_absence.columns]
    if missing_meta:
        raise ValueError(
            f"{gpa_path} is missing expected metadata column(s) {missing_meta}; "
            "cannot reliably tell metadata columns from isolate columns"
        )
    isolate_cols = [c for c in gene_presence_absence.columns if c not in meta_cols]

    # Each row of gene_presence_absence.csv is one cluster; use the row's
    # positional index (0..n_clusters-1) as cluster_id, matching the
    # convention used by the cdhit/mmseqs2/diamond branches above.
    gene_presence_absence = gene_presence_absence.reset_index(drop=True)
    gene_presence_absence.index.name = "cluster_id"

    # Vectorized long-format build (same rationale/performance note as the
    # mmseqs2/diamond crosstab above: avoids an O(n_clusters * n_isolates)
    # nested Python loop, which matters once isolate/cluster counts get
    # into the thousands on real data):
    #   1. melt to (cluster_id, isolate, cell) long format
    #   2. drop empty cells (absent cluster in that isolate -- expected)
    #   3. split each cell on ";"/tab into one or more gene ids (paralogs)
    #      and explode so each gene id gets its own row
    #   4. pd.crosstab -> dense cluster x gene membership matrix
    melted = (
        gene_presence_absence[isolate_cols]
        .reset_index()
        .melt(id_vars="cluster_id", var_name="isolate", value_name="gene_id")
    )
    melted["gene_id"] = melted["gene_id"].astype(str).str.strip()
    # pandas turns empty/NaN cells into the literal string "nan" after
    # astype(str); treat both that and a truly empty string as "absent".
    melted = melted[(melted["gene_id"] != "") & (melted["gene_id"].str.lower() != "nan")]

    melted["gene_id"] = melted["gene_id"].str.split(r"[;\t]")
    melted = melted.explode("gene_id")
    melted["gene_id"] = melted["gene_id"].str.strip()
    melted = melted[melted["gene_id"] != ""]
    melted = melted.reset_index(drop=True)

    if melted.empty:
        # Degenerate but valid case: every cell was empty. Return an
        # all-absent matrix with the right cluster rows and no gene columns.
        outdf = pd.DataFrame(index=gene_presence_absence.index.tolist())
        outdf.index.name = "cluster_id"
        return outdf

    crosstab = pd.crosstab(melted["cluster_id"], melted["gene_id"])
    outdf = crosstab.astype(float)
    outdf[outdf == 0] = -1.0
    outdf = outdf.reindex(index=gene_presence_absence.index.tolist(), fill_value=-1.0)
    outdf.index.name = "cluster_id"
    return outdf



def get_dfs_from_sketch(folderpath, true_max_gene=None):
    print(f"[TRACE] >>> Entering get_dfs_from_sketch() - defined at line 1603 of {__file__}")
    """Parse the sketch/HDBSCAN clustering outputs (one run can contain
    several sub-methods, e.g. hdbscan_dist/hdbscan_tsne/hdbscan_umap, all
    stored together in one clusters.tsv) into the common
    (cluster_id x gene_id) membership-matrix format used elsewhere.

    Input:
        folderpath    -- path to the sketch method's result folder.
        true_max_gene -- highest simulated gene index expected, used to
                          pad each sub-method's matrix so it covers the
                          full gene universe even for genes it dropped.
    Output: dict mapping sub-method name (e.g. "hdbscan_dist") -> a
        (cluster_id x gene_id) DataFrame in the same -1.0/1.0 convention
        as get_df_from_clusterer. Returns {} if the expected clusters.tsv
        file is missing (this sketch run didn't complete/wasn't run).
    """
    tsv_path = os.path.join(folderpath, "distance_clustering", "clusters.tsv")
    if not os.path.isfile(tsv_path):
        return {}
 
    alldf = pd.read_csv(tsv_path, sep="\t")
    # Gene ids in clustering.tsv look like "geneid_0_1" (an extra isoform-style
    # suffix); strip that down to "geneid_0" so it matches the plain
    # "geneid_N" convention used everywhere else in this script.
    alldf["member"] = alldf["member"].apply(
        lambda gid: "_".join(gid.split("_")[:2]) if gid.count("_") >= 2 else gid
    )
    
    outdict = {}
    for method_name, methoddf in alldf.groupby("method"):
        genelist = sorted(
            set(methoddf["member"]), key=lambda x: int(x.split("_")[1])
        )
        gene_nums = {int(g.split("_")[1]) for g in genelist}
 
        max_gene = max(gene_nums) if gene_nums else -1
        if true_max_gene is not None:
            max_gene = max(max_gene, true_max_gene)
 
        # Pad with genes sketch never emitted a row for (e.g. filtered out
        # upstream), same rationale as the panaroo/ppanggolin/panta branches.
        missing = [
            f"geneid_{i}" for i in range(max_gene + 1) if i not in gene_nums
        ]
        if missing:
            genelist = genelist + missing
            genelist.sort(key=lambda x: int(x.split("_")[1]))
 
        cluster_ids = sorted(methoddf["cluster_id"].unique())
        cluster_to_genes = methoddf.groupby("cluster_id")["member"].apply(set)
 
        listoflists = []
        for cluster_index, cid in enumerate(cluster_ids):
            genes_in_cluster = cluster_to_genes[cid]
            row = [cluster_index] + [
                1.0 if gene in genes_in_cluster else -1.0 for gene in genelist
            ]
            listoflists.append(row)
 
        # Every padded/missing gene (dropped by this clusterer) is grouped
        # into one extra "unassigned" pseudo-cluster, so it still shows up
        # in the matrix without being falsely counted as a member of a
        # real cluster.
        if missing:
            missingset = set(missing)
            new_cluster_id = len(cluster_ids)
            row = [new_cluster_id] + [
                1.0 if gene in missingset else -1.0 for gene in genelist
            ]
            listoflists.append(row)
 
        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + genelist)
        outdict[method_name] = outdf.set_index("cluster_id")
 
    return outdict


# === (real-data support) ===================================
def get_dfs_from_sketch_realdata(folderpath):
    print(f"[TRACE] >>> Entering get_dfs_from_sketch_realdata() - defined at line 1681 of {__file__}")
    """Real-data equivalent of get_dfs_from_sketch, following the same
    "restrict to what real data actually needs" rationale as
    get_df_from_clusterer_realdata's docstring.

    Differences from get_dfs_from_sketch, and why:
      1. No "geneid_0_1" -> "geneid_0" isoform-suffix stripping. That
         stripping only existed because the simulation pipeline encodes an
         extra isoform-style suffix onto its synthetic "geneid_N" ids. Real
         gene IDs are already correct as-is (per the real-data run), so the
         "member" column is used unmodified.
      2. No true_max_gene padding, and consequently no extra "unassigned"
         pseudo-cluster for padded genes -- both existed purely to align a
         clusterer's gene universe with the ground-truth gene universe
         (see get_purity), and there is no ground truth for real data.
         Every gene sketch actually emitted a row for is simply used as-is.
      3. Gene lists are sorted lexically rather than by parsing out a
         numeric "geneid_N" suffix, since real gene IDs (locus tags,
         accessions, etc.) have no such guaranteed structure. Sort order
         has no effect on any downstream metric.

    Input:
        folderpath -- path to the sketch method's result folder.
    Output: dict mapping sub-method name (e.g. "hdbscan_dist") -> a
        (cluster_id x gene_id) DataFrame in the same -1.0/1.0 convention
        as get_df_from_clusterer_realdata. Returns {} if the expected
        clusters.tsv file is missing (this sketch run didn't complete/
        wasn't run).
    """
    tsv_path = os.path.join(folderpath, "distance_clustering", "clusters.tsv")
    if not os.path.isfile(tsv_path):
        return {}

    alldf = pd.read_csv(tsv_path, sep="\t")

    allowed_methods = set(CONNECTED_COMPONENTS_METHOD_NAMES) | {"hdbscan_umap"}
    alldf = alldf[alldf["method"].isin(allowed_methods)]

    outdict = {}
    for method_name, methoddf in alldf.groupby("method"):
        genelist = sorted(set(methoddf["member"]))

        cluster_ids = sorted(methoddf["cluster_id"].unique())
        cluster_to_genes = methoddf.groupby("cluster_id")["member"].apply(set)

        listoflists = []
        for cluster_index, cid in enumerate(cluster_ids):
            genes_in_cluster = cluster_to_genes[cid]
            row = [cluster_index] + [
                1.0 if gene in genes_in_cluster else -1.0 for gene in genelist
            ]
            listoflists.append(row)

        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + genelist)
        outdict[method_name] = outdf.set_index("cluster_id")

    return outdict


def get_dfs_from_embeddings(folderpath, true_max_gene=None):
    print(f"[TRACE] >>> Entering get_dfs_from_embeddings() - defined at line 1740 of {__file__}")
    """Embeddings/HDBSCAN equivalent of get_dfs_from_sketch: parses the
    embeddings clustering output (multiple sub-methods in one
    clusters.tsv, e.g. embed_hdbscan_raw/embed_hdbscan_tsne/
    embed_hdbscan_umap) into the common (cluster_id x gene_id)
    membership-matrix format.

    Input/Output: identical shape to get_dfs_from_sketch, except read
        from a different file ("clustering/clusters.tsv" instead of
        "distance_clustering/clusters.tsv") and each output dict key is
        prefixed with "embed_" (e.g. "embed_hdbscan_raw") to distinguish
        it from the plain sketch sub-methods sharing the same base names.
    """
    tsv_path = os.path.join(folderpath, "clustering", "clusters.tsv")
    if not os.path.isfile(tsv_path):
        return {}

    alldf = pd.read_csv(tsv_path, sep="\t")
    # Gene ids in clusters.tsv look like "geneid_0_1" (an extra isoform-style
    # suffix); strip that down to "geneid_0" so it matches the plain
    # "geneid_N" convention used everywhere else in this script.
    alldf["member"] = alldf["member"].apply(
        lambda gid: "_".join(gid.split("_")[:2]) if gid.count("_") >= 2 else gid
    )

    outdict = {}
    for method_name, methoddf in alldf.groupby("method"):
        genelist = sorted(
            set(methoddf["member"]), key=lambda x: int(x.split("_")[1])
        )
        gene_nums = {int(g.split("_")[1]) for g in genelist}

        max_gene = max(gene_nums) if gene_nums else -1
        if true_max_gene is not None:
            max_gene = max(max_gene, true_max_gene)

        # Pad with genes embeddings/HDBSCAN never emitted a row for (e.g.
        # filtered out upstream), same rationale as the sketch branch.
        missing = [
            f"geneid_{i}" for i in range(max_gene + 1) if i not in gene_nums
        ]
        if missing:
            genelist = genelist + missing
            genelist.sort(key=lambda x: int(x.split("_")[1]))

        cluster_ids = sorted(methoddf["cluster_id"].unique())
        cluster_to_genes = methoddf.groupby("cluster_id")["member"].apply(set)

        listoflists = []
        for cluster_index, cid in enumerate(cluster_ids):
            genes_in_cluster = cluster_to_genes[cid]
            row = [cluster_index] + [
                1.0 if gene in genes_in_cluster else -1.0 for gene in genelist
            ]
            listoflists.append(row)

        # As above: dropped/padded genes are grouped into one extra
        # pseudo-cluster rather than silently mixed into a real cluster.
        if missing:
            missingset = set(missing)
            new_cluster_id = len(cluster_ids)
            row = [new_cluster_id] + [
                1.0 if gene in missingset else -1.0 for gene in genelist
            ]
            listoflists.append(row)

        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + genelist)
        outdict[f"embed_{method_name}"] = outdf.set_index("cluster_id")

    return outdict


def count_singleton_clusters(thedf):
    print(f"[TRACE] >>> Entering count_singleton_clusters() - defined at line 1812 of {__file__}")
    """Count clusters that contain exactly one gene ("orphan" genes with
    no detected homologues).

    Input:  thedf -- a (cluster_id x gene_id) membership matrix (cells
            >= 0 mean membership, see get_df_from_clusterer).
    Output: int, the number of rows (clusters) with exactly one member.
    Biological reading: a high singleton count usually signals either a
    genuinely large accessory/strain-specific gene pool, or overly
    strict clustering (e.g. too high a sequence-identity threshold)
    artificially splitting true gene families apart.
    """
    
    member_counts = (thedf >= 0.0).sum(axis=1)
    return int((member_counts == 1).sum())

def count_pairs_clusters(thedf):
    print(f"[TRACE] >>> Entering count_pairs_clusters() - defined at line 1828 of {__file__}")
    """Count clusters that contain exactly two genes.

    Input:  thedf -- a (cluster_id x gene_id) membership matrix.
    Output: int, the number of clusters with exactly 2 members. Together
    with count_singleton_clusters, this is used (as a stacked bar) to show
    how the cluster-size distribution shifts between methods/parameters --
    e.g. an inflated pair count alongside singletons can indicate a
    clusterer that is splitting true multi-member gene families into
    many small fragments.
    """
    member_counts = (thedf >= 0.0).sum(axis=1)
    return int((member_counts == 2).sum())

def get_time_diff_from_file(inpath):
    print(f"[TRACE] >>> Entering get_time_diff_from_file() - defined at line 1842 of {__file__}")
    """Parse a "timebenchmark.txt"-style file containing one line with a
    "<start_time>=>[<end_time>]" timestamp pair, and return the elapsed
    wall-clock runtime in seconds.

    Input:  inpath -- path to the timing file (format
            "%d/%m/%Y-%H:%M:%S=>%d/%m/%Y-%H:%M:%S").
    Output: float, runtime in seconds (end - start), used to populate the
    "runtime" column plotted by plotter/plotter_pointplots.
    """
    time0 = None
    time1 = None
    with open(inpath, "r") as f:
        for line in f:
            if "=>" in line:
                splits = line.strip().split("=>")
                time0 = datetime.strptime(splits[0], "%d/%m/%Y-%H:%M:%S")
                time1 = datetime.strptime(splits[1], "%d/%m/%Y-%H:%M:%S")
                break
    return (time1 - time0).total_seconds()


def parse_time_per_method_file(inpath):
    print(f"[TRACE] >>> Entering parse_time_per_method_file() - defined at line 1864 of {__file__}")
    """Parses a time_per_method.txt file such as:
        load_distance_matrix: 2690.072s sweep_total: 15.786s tsne_embedding: 29.435s
        umap_embedding: 8.882s hdbscan_dist_fit: 0.787s hdbscan_tsne_fit: 0.106s
        hdbscan_umap_fit: 0.105s method_hdbscan_dist_total: 0.787s
        method_hdbscan_tsne_total: 29.542s method_hdbscan_umap_total: 8.986s
        plotting: 11.166s stats_and_tsv: 0.367s total: 2756.783s
    Returns a dict mapping each key (e.g. "method_hdbscan_tsne_total") to its
    value in seconds (float). Works regardless of whether entries are spread
    over multiple lines or all on one line.
    """
    outdict = {}
    with open(inpath, "r") as f:
        content = f.read()
    for key, value in re.findall(r"(\S+):\s*([\d.]+)s", content):
        outdict[key] = float(value)
    return outdict


def get_species_name(inpath):
    print(f"[TRACE] >>> Entering get_species_name() - defined at line 1883 of {__file__}")
    """Read the first line of a species-name file (one plain-text line
    naming the simulated/real organism for this assembly).

    Input:  inpath -- path to the species-name text file.
    Output: str, the species name with surrounding whitespace stripped,
    or "" if the file is empty. Used to build namedict for plot titles.
    """
    with open(inpath, "r") as f:
        for line in f:
            return line.strip()
    return ""


def check_status_of_folder(clusterer, path):
    print(f"[TRACE] >>> Entering check_status_of_folder() - defined at line 1897 of {__file__}")
    """Sanity-check that a clustering tool actually produced its expected
    main output file in a given result folder, before we attempt to parse
    it (avoids crashing deep inside a parser on a partially-run/failed job).

    Input:
        clusterer -- name of the clustering tool (see CLUSTERERS).
        path      -- path to that tool's result folder to check.
    Output: bool, True if the tool-specific expected output file exists
        under `path` (e.g. "cdhit.clstr" for CD-HIT, "diamond" for
        DIAMOND, ...), False otherwise (also prints a diagnostic message
        in both the "unknown clusterer" and "missing file" cases).
    """
    if clusterer == "cdhit":
        filenam = "cdhit.clstr"
    elif clusterer == "mmseqs2":
        filenam = "mmseqs2_cluster.tsv"
    elif clusterer == "diamond":
        filenam = "diamond"
    elif clusterer == "panta":
        filenam = "panta/annotated_clusters.json"
    elif clusterer == "panaroo":
        filenam = "panaroo/combined_protein_cdhit_out.txt.clstr"
    elif clusterer == "ppanggolin":
        filenam = "ppanggolin/pangenome.h5"
    elif clusterer == "sketch":
        filenam = "distance_clustering/clusters.tsv"
    elif clusterer == "embeddings":
        filenam = "clustering/clusters.tsv"
    elif clusterer == "panx":
        filenam = "protein_faa/diamond_matches/allclusters_final.tsv"
    else:
        print("Invalid clusterer " + clusterer)
        return False
    checkpath = os.path.join(path, filenam)
    if os.path.isfile(checkpath):
        return True
    print("Invalid file " + checkpath)
    return False


def get_truth_matrix_path(datapath, assembly, seed):
    print(f"[TRACE] >>> Entering get_truth_matrix_path() - defined at line 1938 of {__file__}")
    """Locate the single ground-truth cluster-membership file for one
    simulated assembly/seed (simulation pipeline only), by looking for
    the one file in that seed's data directory whose name contains
    "truth_matrix".

    Input:
        datapath -- root path to the simulation data.
        assembly -- assembly identifier (subdirectory name).
        seed     -- random-seed identifier (subdirectory name).
    Output: str, full path to the truth-matrix file.
    Raises: (implicitly, via len(matches) check just below) if zero or
        more than one candidate file is found, since the ground truth
        must be unambiguous.
    """
    truth_seed_dir = os.path.join(datapath, "simulations", str(assembly), str(seed))
    matches = [
        os.path.join(truth_seed_dir, el)
        for el in os.listdir(truth_seed_dir)
        if "truth_matrix" in el
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one truth matrix in {truth_seed_dir}, found {len(matches)}"
        )
    return matches[0]


def get_simulation_ground_truth_cluster_stats(datapath, assembly):
    print(f"[TRACE] >>> Entering get_simulation_ground_truth_cluster_stats()")
    """Ground-truth number-of-clusters summary (n_seeds/mean/min/max) for
    one simulated assembly, computed the same way as
    fast_count_groundtruth_clusters.sh: for every seed directory under
    datapath/simulations/<assembly>/, locate that seed's single
    *truth_matrix* file and count the number of DISTINCT values in its
    "original_gene" column (the number of true simulated gene families),
    then take the mean/min/max of that per-seed count across all seeds
    found.

    This reuses get_truth_matrix_path() -- the exact same lookup used by
    get_info_from_folder() to load the ground truth for the metrics
    themselves -- so the set of files considered here is identical to
    the one used everywhere else in the pipeline; only the aggregation
    (mean/min/max across seeds) is new, and it mirrors the bash script's
    awk-based distinct-value count 1:1 (nunique() on "original_gene" is
    the same quantity as the bash script's `length(seen)` over that
    column).

    Input:
        datapath -- root path to the simulation data (same as
            args.datapath / get_truth_matrix_path's datapath argument).
        assembly -- assembly identifier (subdirectory name).
    Output: dict with keys "n_seeds", "mean", "min", "max", or None if
        no seed under this assembly has a locatable truth_matrix file
        (e.g. assembly not present under datapath, mirroring the bash
        script's "no truth_matrix files found" case for that assembly).
    """
    assembly_dir = os.path.join(datapath, "simulations", str(assembly))
    if not os.path.isdir(assembly_dir):
        warnings.warn(
            f"Cannot compute ground-truth cluster-count stats for assembly "
            f"{assembly!r}: {assembly_dir} does not exist",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    n_clusters_per_seed = []
    for seed in sorted(os.listdir(assembly_dir)):
        seed_dir = os.path.join(assembly_dir, seed)
        if not os.path.isdir(seed_dir):
            continue
        try:
            truth_path = get_truth_matrix_path(datapath, assembly, seed)
        except RuntimeError as exc:
            warnings.warn(
                f"Skipping ground-truth cluster-count stats for "
                f"{assembly}/{seed}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        truthmatrix = pd.read_csv(truth_path, sep="\t")
        n_clusters_per_seed.append(int(truthmatrix["original_gene"].nunique()))

    if not n_clusters_per_seed:
        warnings.warn(
            f"No truth_matrix files found for assembly {assembly!r} under "
            f"{assembly_dir}; skipping ground-truth cluster-count overlay",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    arr = np.array(n_clusters_per_seed, dtype=float)
    return {
        "n_seeds": len(n_clusters_per_seed),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def get_info_from_folder(theargs):
    print(f"[TRACE] >>> Entering get_info_from_folder() - defined at line 1966 of {__file__}")
    """Top-level orchestration function for one (assembly, seed) unit of
    work in the SIMULATION pipeline: loads the ground truth, then runs
    every requested clusterer's output through get_df_from_clusterer /
    calculate_values_from_cluster_matrix, and collects everything needed
    downstream (metric rows, per-method label dicts, gene counts).

    Input:  theargs -- (thedir, theass, theseed, datapath) tuple, where
        thedir is this seed's result directory, theass/theseed are the
        assembly/seed identifiers, and datapath is the root simulation
        data path (needed to locate the ground-truth truth_matrix file).
    Output: a tuple analogous to get_info_from_folder_realdata's return
        value -- (listoflists, theass, nameofass, theseed,
        method_labels_out, n_genes_seen) -- consumed by main() to build
        outdf and the various per-assembly lookup dicts
        (method_labels_by_assembly, total_genes_by_assembly, ...).
    """
    thedir, theass, theseed, datapath = theargs
    truthpath = get_truth_matrix_path(datapath, theass, theseed)
    truthmatrix = pd.read_csv(truthpath, sep="\t")
    truthmatrix = truthmatrix.set_index("gene_id")
    truthlabels = list(truthmatrix["original_gene"])
    # One-hot encode the ground-truth cluster assignment (one boolean
    # column per true gene family) to build the truthdf format expected
    # by get_purity/calculate_values_from_cluster_matrix.
    one_hot = pd.get_dummies(truthmatrix["original_gene"])
    truthdf = truthmatrix.drop("original_gene", axis=1)
    truthdf = truthdf.join(one_hot)

    # True upper bound on gene numbering, taken from the ground truth rather
    # than from any individual clusterer's output. Any clusterer's gene list
    # must be padded at least up to this number, or get_purity's lookup
    # (which indexes by gene number straight from truthdf) can go out of range.
    true_max_gene = max(
        int(re.search(r"geneid_(\d+)", g).group(1)) for g in truthdf.index
    )

    # === CHANGE (gene-deletion penalty, requirements 1 & 2) ===
    # Total number of genes in the ORIGINAL dataset for this assembly/seed.
    # This is the reference/universe used later to (a) penalise the pairwise
    # gene-retention F1 score for genes a method deleted, and (b) compute the
    # per-method/per-seed "percentage of genes deleted" (see
    # compute_gene_deletion_dataframe / build_pairwise_f1_matrix below).
    n_original_genes = len(truthdf.index)
    print(f"\t- Getting information from {thedir} execution, {theass} assembly, and {theseed} seed")

    speciesfile = os.path.join(thedir, str(theass), "assembly_species.txt")
    if os.path.isfile(speciesfile):
        nameofass = get_species_name(speciesfile)
    else:
        nameofass = ""

    listoflists = []
    # combo ("clusterer/seqtype", e.g. "mmseqs2/nt") -> {gene_id: cluster_label}
    # captured here (at the default c for the c-swept clusterers) so we can
    # later compute pairwise ARI between methods themselves, not just vs
    # ground truth. See build_pairwise_ari_matrix / plot_pairwise_ari_heatmap.
    method_labels_out = {}
    seed_result_dir = os.path.join(thedir, str(theass), str(theseed))
    for folder_name in os.listdir(seed_result_dir):
        folderpath = os.path.join(seed_result_dir, folder_name)
        if not os.path.isdir(folderpath):
            continue

        splits = folder_name.split("_")
        tmpclusterer = splits[0]
        if tmpclusterer not in CLUSTERERS:
            warnings.warn(
                f"Skipping non-clusterer folder {folderpath}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        try:
            paramdict = get_param_dict_from_splits(splits[1:]) if len(splits) > 1 else {}
        except ValueError as exc:
            warnings.warn(
                f"Skipping malformed result folder {folderpath}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        invalid_params = [key for key in paramdict if key not in DEFAULT_PARAMS]
        if invalid_params:
            warnings.warn(
                f"Skipping result folder {folderpath}; unsupported parameters {invalid_params}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        tmpseqtype = paramdict.get("st", "aa" if tmpclusterer in ("panaroo", "ppanggolin", "panta", "panx", "embeddings") else DEFAULT_PARAMS["st"])
        if tmpclusterer == "diamond" and tmpseqtype == "nt":
            warnings.warn(
                f"Skipping disabled diamond+nt result folder {folderpath}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if tmpclusterer == "panaroo" and tmpseqtype == "nt":
            warnings.warn(
                f"Skipping disabled panaroo+nt result folder {folderpath}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if tmpclusterer == "panta" and tmpseqtype == "nt":
            warnings.warn(
                f"Skipping disabled panta+nt result folder {folderpath}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if tmpclusterer == "panx" and tmpseqtype == "nt":
            warnings.warn(
                f"Skipping disabled panx+nt result folder {folderpath}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if tmpclusterer == "embeddings" and tmpseqtype == "nt":
            warnings.warn(
                f"Skipping disabled embeddings+nt result folder {folderpath}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if not check_status_of_folder(tmpclusterer, folderpath):
            continue

        if tmpclusterer == "sketch":
            sketch_dfs = get_dfs_from_sketch(folderpath, true_max_gene)
            if not sketch_dfs:
                warnings.warn(
                    f"No sketch clusters.tsv files found in {folderpath}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            runtime = get_time_diff_from_file(os.path.join(folderpath, "timebenchmark.txt"))

            # Per-method runtimes (load_distance_matrix, embeddings, hdbscan fit, ...)
            # are broken down in distance_clustering/time_per_method.txt; use the
            # "method_<name>_total" entry for each sketch/HDBSCAN method instead of
            # the single whole-folder runtime, so each method's plotted runtime
            # reflects the time actually spent on that specific method (embedding +
            # hdbscan fit) rather than the total sketch folder runtime.
            time_per_method_path = os.path.join(folderpath, "distance_clustering", "time_per_method.txt")
            method_runtimes = {}
            if os.path.isfile(time_per_method_path):
                method_runtimes = parse_time_per_method_file(time_per_method_path)

            for method_name, thedf in sketch_dfs.items():
                n_clusters = len(thedf.index)
                n_singletons = count_singleton_clusters(thedf)
                n_pairs = count_pairs_clusters(thedf)
                
                paramlist = [
                    tmpseqtype if el == "st" else DEFAULT_PARAMS[el]
                    for el in PARAMORDER
                ]
                method_runtime = method_runtimes.get(
                    f"method_{method_name}_total", runtime
                )
                listoflists.append(
                    calculate_values_from_cluster_matrix(
                        (theass, theseed, method_name), thedf, truthlabels, truthdf
                    )
                    + [n_clusters, n_singletons, n_pairs]
                    + paramlist
                    + [method_runtime]
                )
                # Sketch/HDBSCAN methods have no c sweep, so always capture them.
                genes_i, labels_i = get_labels_list_from_df(thedf)
                method_labels_out[f"{method_name}/{tmpseqtype}"] = dict(zip(genes_i, labels_i))
            continue

        if tmpclusterer == "embeddings":
            embed_dfs = get_dfs_from_embeddings(folderpath, true_max_gene)
            if not embed_dfs:
                warnings.warn(
                    f"No embeddings clusters.tsv files found in {folderpath}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            runtime = get_time_diff_from_file(os.path.join(folderpath, "timebenchmark.txt"))

            # Same rationale as sketch: prefer the per-method breakdown from
            # clustering/time_per_method.txt (embedding + hdbscan fit time for
            # that specific method) over the whole-folder runtime, when available.
            time_per_method_path = os.path.join(folderpath, "clustering", "time_per_method.txt")
            method_runtimes = {}
            if os.path.isfile(time_per_method_path):
                method_runtimes = parse_time_per_method_file(time_per_method_path)

            for method_name, thedf in embed_dfs.items():
                n_clusters = len(thedf.index)
                n_singletons = count_singleton_clusters(thedf)
                n_pairs = count_pairs_clusters(thedf)

                paramlist = [
                    tmpseqtype if el == "st" else DEFAULT_PARAMS[el]
                    for el in PARAMORDER
                ]
                # method_runtimes is keyed by the raw (un-prefixed) method name,
                # e.g. "method_hdbscan_dist_total", since that's what
                # time_per_method.txt records; method_name here is the
                # "embed_"-prefixed dict key.
                raw_method_name = method_name[len("embed_"):]
                method_runtime = method_runtimes.get(
                    f"method_{raw_method_name}_total", runtime
                )
                listoflists.append(
                    calculate_values_from_cluster_matrix(
                        (theass, theseed, method_name), thedf, truthlabels, truthdf
                    )
                    + [n_clusters, n_singletons, n_pairs]
                    + paramlist
                    + [method_runtime]
                )
                # Embeddings/HDBSCAN methods have no c sweep, so always capture them.
                genes_i, labels_i = get_labels_list_from_df(thedf)
                method_labels_out[f"{method_name}/{tmpseqtype}"] = dict(zip(genes_i, labels_i))
            continue

        thedf = get_df_from_clusterer(tmpclusterer, folderpath, true_max_gene)
        runtime = get_time_diff_from_file(os.path.join(folderpath, "timebenchmark.txt"))
        n_clusters = len(thedf.index)
        n_singletons = count_singleton_clusters(thedf)
        n_pairs = count_pairs_clusters(thedf)
        paramlist = [
            paramdict[el] if el in paramdict else (tmpseqtype if el == "st" else DEFAULT_PARAMS[el])
            for el in PARAMORDER
        ]
        listoflists.append(
            calculate_values_from_cluster_matrix(
                (theass, theseed, tmpclusterer), thedf, truthlabels, truthdf
            )
            + [n_clusters, n_singletons, n_pairs]
            + paramlist
            + [runtime]
        )
        # These clusterers are swept over c; only keep the default-c run so
        # the pairwise comparison lines up with the vs-truth heatmap (which
        # also filters to c == DEFAULT_PARAMS["c"]).
        c_value = paramlist[PARAMORDER.index("c")]
        if c_value == DEFAULT_PARAMS["c"]:
            genes_i, labels_i = get_labels_list_from_df(thedf)
            method_labels_out[f"{tmpclusterer}/{tmpseqtype}"] = dict(zip(genes_i, labels_i))
    if not listoflists:
        warnings.warn(
            f"No valid clustering outputs found for {theass}/{theseed}; skipping",
            RuntimeWarning,
            stacklevel=2,
        )
    # === CHANGE: added n_original_genes as a 6th tuple element (requirement 1 & 2) ===
    return (listoflists, theass, nameofass, theseed, method_labels_out, n_original_genes)


def is_refound_gene_id(gene_id):
    print(f"[TRACE] >>> Entering is_refound_gene_id() - defined at line 2227 of {__file__}")
    """True if `gene_id` looks like a Panaroo-added "refound" gene id
    (Panaroo encodes this directly in the gene id string it writes into
    gene_presence_absence.csv, e.g. "NLEIDEKG_01145_refound_1")."""
    return "refound" in str(gene_id)


def filter_refound_genes(genes, labels):
    print(f"[TRACE] >>> Entering filter_refound_genes() - defined at line 2234 of {__file__}")
    """Real-data-only filter: given parallel (genes, labels) lists as
    returned by get_labels_list_from_df, drop every entry whose gene id is
    a Panaroo "refound" gene (see is_refound_gene_id), and report how many
    were removed.

    This is the real-data analogue of the exclusion already applied on the
    simulation/gene_data.csv side (see get_df_from_clusterer_realdata's old
    id_map construction and get_realdata_reference_gene_set): refound genes
    are Panaroo-reconstructed sequences used to patch annotation gaps, not
    genes that were genuinely observed and clustered, so they must not be
    allowed to influence any agreement-with-another-method metric (AMI,
    ARI, purity, v-measure, or the pairwise F1/Dice score). They are,
    however, still real entries in Panaroo's *output*, so they are kept
    for the separate "% of genes added by Panaroo" statistic instead (see
    compute_gene_addition_dataframe).

    Returns (filtered_genes, filtered_labels, n_refound_removed).
    """
    filtered_genes = []
    filtered_labels = []
    n_refound = 0
    for gene, label in zip(genes, labels):
        if is_refound_gene_id(gene):
            n_refound += 1
            continue
        filtered_genes.append(gene)
        filtered_labels.append(label)
    return filtered_genes, filtered_labels, n_refound


def _process_one_realdata_folder(args):
    print(f"[TRACE] >>> Entering _process_one_realdata_folder() - defined at line 2265 of {__file__}")
    """Worker for a single clusterer output folder in real-data mode
    (e.g. "mmseqs2_st-aa_c-0.9/"). Pulled out of get_info_from_folder_realdata
    so it can be dispatched to a multiprocessing Pool -- see the
    parallelization note in get_info_from_folder_realdata's docstring.

    Returns None if this folder should be skipped (wrong clusterer, bad
    params, disabled combo, missing expected output file), otherwise a
    LIST of result dicts (everything the caller needs to fold into its
    accumulators) -- a list rather than a single dict because, like the
    simulation "sketch" branch in get_info_from_folder, one sketch result
    folder can contain several sub-methods (hdbscan_dist/hdbscan_tsne/
    hdbscan_umap) sharing one clusters.tsv, each of which needs its own
    row/combo_key. Every other (single-method) clusterer still returns a
    one-element list, so callers can always just flatten and iterate.
    """
    folderpath, folder_name, theass, theseed = args

    splits = folder_name.split("_")
    tmpclusterer = splits[0]
    if tmpclusterer not in REAL_DATA_CLUSTERERS:
        return None

    try:
        paramdict = get_param_dict_from_splits(splits[1:]) if len(splits) > 1 else {}
    except ValueError as exc:
        warnings.warn(
            f"Skipping malformed result folder {folderpath}: {exc}",
            RuntimeWarning, stacklevel=2,
        )
        return None

    invalid_params = [key for key in paramdict if key not in DEFAULT_PARAMS]
    if invalid_params:
        warnings.warn(
            f"Skipping result folder {folderpath}; unsupported parameters {invalid_params}",
            RuntimeWarning, stacklevel=2,
        )
        return None

    # panaroo has no nt variant (same default as the simulations path -- see
    # get_info_from_folder's equivalent tmpseqtype line), so default it to
    # "aa" rather than DEFAULT_PARAMS["st"] ("nt") when "st" isn't in the
    # folder name.
    tmpseqtype = paramdict.get("st", "aa" if tmpclusterer in ("panaroo", "ppanggolin", "panta", "panx") else DEFAULT_PARAMS["st"])
    if tmpclusterer == "diamond" and tmpseqtype == "nt":
        warnings.warn(
            f"Skipping disabled diamond+nt result folder {folderpath}",
            RuntimeWarning, stacklevel=2,
        )
        return None
    if tmpclusterer == "panaroo" and tmpseqtype == "nt":
        warnings.warn(
            f"Skipping disabled panaroo+nt result folder {folderpath}",
            RuntimeWarning, stacklevel=2,
        )
        return None
    if tmpclusterer == "panta" and tmpseqtype == "nt":
        warnings.warn(
            f"Skipping disabled panta+nt result folder {folderpath}",
            RuntimeWarning, stacklevel=2,
        )
        return None
    if tmpclusterer == "panx" and tmpseqtype == "nt":
        warnings.warn(
            f"Skipping disabled panx+nt result folder {folderpath}",
            RuntimeWarning, stacklevel=2,
        )
        return None
    if not check_status_of_folder(tmpclusterer, folderpath):
        return None

    paramlist = [
        paramdict[el] if el in paramdict else (tmpseqtype if el == "st" else DEFAULT_PARAMS[el])
        for el in PARAMORDER
    ]
    c_value = paramlist[PARAMORDER.index("c")]

    if tmpclusterer == "sketch":
        print(f"\t\t- Reading sketch ({tmpseqtype}) from {folderpath} ...")
        sketch_dfs = get_dfs_from_sketch_realdata(folderpath)
        if not sketch_dfs:
            warnings.warn(
                f"No sketch clusters.tsv files found in {folderpath}",
                RuntimeWarning, stacklevel=2,
            )
            return None

        runtime_path = os.path.join(folderpath, "timebenchmark.txt")
        runtime = get_time_diff_from_file(runtime_path) if os.path.isfile(runtime_path) else np.nan

        time_per_method_path = os.path.join(folderpath, "distance_clustering", "time_per_method.txt")
        method_runtimes = {}
        if os.path.isfile(time_per_method_path):
            method_runtimes = parse_time_per_method_file(time_per_method_path)

        results = []
        for method_name, thedf in sketch_dfs.items():
            n_clusters = len(thedf.index)
            n_singletons = count_singleton_clusters(thedf)
            n_pairs = count_pairs_clusters(thedf)
            method_runtime = method_runtimes.get(f"method_{method_name}_total", runtime)
            print(
                f"\t\t  done: {method_name} ({tmpseqtype}) -> "
                f"{n_clusters} clusters, {len(thedf.columns)} genes, "
                f"runtime={method_runtime if not (isinstance(method_runtime, float) and np.isnan(method_runtime)) else 'n/a'}s"
            )

            # 8 truth-dependent NaNs: adj_rand_index, adj_rand_index_pvalue,
            # purity, adj_mutual_info, adj_mutual_info_pvalue, homogeneity,
            # completeness, v_measure -- real data has no ground truth, so
            # none of these (including the ARI/AMI p-values) are
            # meaningful here; row shape must still match
            # calculate_values_from_cluster_matrix's output (see
            # build_results_dataframe) so no per-mode branching is needed
            # downstream.
            row = (
                [False, theass, theseed, method_name,
                 np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                + [n_clusters, n_singletons, n_pairs]
                + paramlist
                + [method_runtime]
            )

            genes_i, labels_i = get_labels_list_from_df(thedf)

            combo_key = f"{method_name}/{tmpseqtype}"

            results.append({
                "row": row,
                "genes": genes_i,
                "labels": labels_i,
                "combo_key": combo_key,
                "n_refound": 0,
            })
        return results

    # === Per-method progress prints (real-data mode) ===
    # Printed from whichever process (main, or a Pool worker) handles this
    # folder -- with -j > 1 these interleave across methods running at the
    # same time, which is expected and fine; each line is self-contained.
    print(f"\t\t- Reading {tmpclusterer} ({tmpseqtype}) from {folderpath} ...")
    thedf = get_df_from_clusterer_realdata(tmpclusterer, folderpath)
    runtime_path = os.path.join(folderpath, "timebenchmark.txt")
    runtime = get_time_diff_from_file(runtime_path) if os.path.isfile(runtime_path) else np.nan
    n_clusters = len(thedf.index)
    n_singletons = count_singleton_clusters(thedf)
    n_pairs = count_pairs_clusters(thedf)
    print(
        f"\t\t  done: {tmpclusterer} ({tmpseqtype}) -> "
        f"{n_clusters} clusters, {len(thedf.columns)} genes, "
        f"runtime={runtime if not np.isnan(runtime) else 'n/a'}s"
    )

    # Truth-dependent columns (ARI, ARI p-value, purity, AMI, AMI p-value,
    # homogeneity, completeness, v-measure) do not exist for real data ->
    # NaN (including the ARI/AMI p-values, which are only meaningful when
    # there's a ground truth to permutation-test against), but the row
    # shape is kept identical to calculate_values_from_cluster_matrix's
    # output (8 truth-dependent columns: adj_rand_index,
    # adj_rand_index_pvalue, purity, adj_mutual_info,
    # adj_mutual_info_pvalue, homogeneity, completeness, v_measure) so
    # build_results_dataframe needs no per-mode branching.
    row = (
        [False, theass, theseed, tmpclusterer,
         np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        + [n_clusters, n_singletons, n_pairs]
        + paramlist
        + [runtime]
    )

    genes_i, labels_i = get_labels_list_from_df(thedf)

    if tmpclusterer == "panaroo":
        genes_i, labels_i, n_refound = filter_refound_genes(genes_i, labels_i)
    else:
        n_refound = 0

    combo_key = f"{tmpclusterer}/{tmpseqtype}" if c_value == DEFAULT_PARAMS["c"] else None

    return [{
        "row": row,
        "genes": genes_i,
        "labels": labels_i,
        "combo_key": combo_key,
        "n_refound": n_refound,
    }]


def get_info_from_folder_realdata(theargs, nthreads=1):
    print(f"[TRACE] >>> Entering get_info_from_folder_realdata() - defined at line 2444 of {__file__}")
    """Real-data equivalent of get_info_from_folder.

    === flat layout, no assembly/seed subdirectories ===========
    Unlike the simulations layout (runfolder/simulations/<assembly>/<seed>/
    <clusterer_paramdir>/...), the real_data layout you actually have is
    flat: runfolder/real_data/<clusterer_paramdir>/... directly (e.g.
    "diamond_st-aa_c-0.9/diamond"), with no per-assembly or per-seed
    nesting at all. There is exactly one dataset and one run.

    To reuse the rest of the pipeline unchanged (it is built throughout
    around a MultiIndex of assembly + seed, e.g. for the pairwise-heatmap
    "average over seeds" logic, and for namedict/outfolder plot titles), we
    treat this single flat directory as one pseudo-"assembly" (named
    "real_data") containing one pseudo-"seed" (named "run"). This is purely
    a bookkeeping label -- it does not imply multiple seeds/replicates
    exist; every plot will simply show a single data point per method.

    ==========================================
    discover_analysis_tasks_realdata only ever returns ONE outer task (this
    whole function call), since there's no assembly/seed dimension to split
    across workers the way the simulations path does. So running this
    *whole function* inside main()'s outer Pool.map (as the simulations
    path does) would waste `-j` on a list of length 1, and the real,
    independent parallelizable work -- reading each clusterer's output
    folder (cdhit / mmseqs2 / diamond, and any parameter sweep across
    multiple c/seqtype values within each) -- would still run serially
    inside a single worker.

    Instead, this function does its OWN internal parallelization: it lists
    all candidate result folders first, then farms each one out to
    `_process_one_realdata_folder` via a multiprocessing Pool sized by
    `nthreads` (falling back to a plain for-loop when nthreads <= 1, so
    behaviour with the default -j 1 is unchanged and single-process for
    easy debugging). main() calls this function directly (not through its
    own outer Pool) in real-data mode and passes args.nthreads through, so
    `-j` now actually parallelizes per-method file reading.

    Other differences from the simulation version, and why:
      - No truth matrix is loaded (get_truth_matrix_path/get_purity/
        calculate_values_from_cluster_matrix are never called), since real
        biological data has no ground-truth cluster assignments. The
        truth-dependent metric columns (adj_rand_index, purity,
        adj_mutual_info, homogeneity, completeness, v_measure) are filled
        with NaN so the returned row still matches the column layout
        build_results_dataframe expects -- this lets us reuse that
        function, and any plot that only reads runtime/n_clusters/params,
        completely unchanged.
      - Only CD-HIT, MMseqs2 and DIAMOND folders are processed (requirement
        5); every other clusterer's output folder is ignored even if
        present on disk.
      - Gene tables come from get_df_from_clusterer_realdata, which makes no
        "geneid_N" assumptions (requirement re: gene ID format) and needs no
        true_max_gene padding (there is no ground-truth gene universe).
      - "n_original_genes" here is NOT a ground-truth count (none exists);
        it is simply the number of distinct genes seen across this run's
        kept clustering outputs, kept only for descriptive/logging purposes.
    """
    thedir, theass, theseed, _datapath = theargs
    print(
        f"\t- Getting information from {thedir} (real data, no ground truth, "
        "no assembly/seed nesting)"
    )

    nameofass = "Real data"

    seed_result_dir = thedir
    folder_tasks = [
        (os.path.join(seed_result_dir, folder_name), folder_name, theass, theseed)
        for folder_name in os.listdir(seed_result_dir)
        if os.path.isdir(os.path.join(seed_result_dir, folder_name))
    ]

    if nthreads <= 1:
        raw_results = [_process_one_realdata_folder(t) for t in folder_tasks]
    else:
        with Pool(min(nthreads, max(1, len(folder_tasks)))) as pool:
            raw_results = pool.map(_process_one_realdata_folder, folder_tasks)

    listoflists = []
    method_labels_out = {}

    n_refound_out = {}
    all_genes_seen = set()

    for folder_results in raw_results:
        if not folder_results:
            continue
        for result in folder_results:
            listoflists.append(result["row"])
            all_genes_seen.update(result["genes"])
            if result["combo_key"] is not None:
                method_labels_out[result["combo_key"]] = dict(zip(result["genes"], result["labels"]))
                n_refound_out[result["combo_key"]] = result.get("n_refound", 0)

    if not listoflists:
        warnings.warn(
            f"No valid real-data clustering outputs found for {theass}/{theseed}; skipping",
            RuntimeWarning, stacklevel=2,
        )

    n_genes_seen = len(all_genes_seen)
    return (listoflists, theass, nameofass, theseed, method_labels_out, n_genes_seen, n_refound_out)


# plot idea : how does this metric vary with the clustering threshold, averaged over random seeds, for this assembly

def build_ordered_combo_list(subdf, name=None):
    print(f"[TRACE] >>> Entering build_ordered_combo_list() - defined at line 2551 of {__file__}")
    """Return the clusterer/seqtype combos present in subdf, in the nice,
    consistent COMBO_ORDER order (rather than whatever order set() happens
    to give), skipping combos with no data and combos missing from
    FANCYDICT/CONFIGDICT_COLOURS."""
    x = []
    for combo in COMBO_ORDER:
        clusterer, seqtype = combo.split("/")
        if combo not in FANCYDICT or combo not in CONFIGDICT_COLOURS:
            continue
        mask = (
            (subdf.index.get_level_values("clusterer") == clusterer)
            & (subdf["st"] == seqtype)
        )
        tmpdf = subdf[mask]
        if name is not None:
            tmpdf = tmpdf[name]
        if len(tmpdf):
            x.append(combo)
    return x


def get_family_footnote(x):
    print(f"[TRACE] >>> Entering get_family_footnote() - defined at line 2573 of {__file__}")
    """Return the combined footnote text for whichever method families
    (sketch, embeddings, ...) are present among the combos in x, or None if
    none are present."""
    lines = [
        FAMILY_FOOTNOTES[label]
        for label, method_names in FAMILY_METHOD_NAMES.items()
        if any(value.split("/")[0] in method_names for value in x)
    ]
    if not lines:
        return None
    return "\n".join(lines)


def add_sketch_bracket(ax, x, positions, bar_width=0.5, y_top=-0.28, y_bottom=-0.30,
                        fontprops=None, fontsize=None, row_gap=0.06):
    print(f"[TRACE] >>> Entering add_sketch_bracket() - defined at line 2587 of {__file__}")
    """Draw a small bracket + label under each method family's columns (sketch,
    embeddings, ...), so readers can see at a glance that they're one family of
    methods. If more than one family is present, brackets are stacked below
    each other so they don't overlap. Kept under its original name for
    backwards compatibility with existing call sites."""
    row = 0
    for label, method_names in FAMILY_METHOD_NAMES.items():
        family_idx = [i for i, val in enumerate(x) if val.split("/")[0] in method_names]
        if not family_idx:
            continue
        lo, hi = min(family_idx), max(family_idx)
        x0 = positions[lo] - bar_width / 2
        x1 = positions[hi] + bar_width / 2
        row_y_top = y_top - row * row_gap
        row_y_bottom = y_bottom - row * row_gap
        trans = ax.get_xaxis_transform()  # x in data coords, y in axes-fraction coords
        ax.plot(
            [x0, x0, x1, x1], [row_y_top, row_y_bottom, row_y_bottom, row_y_top],
            transform=trans, color="black", linewidth=0.8, clip_on=False,
        )
        ax.text(
            (x0 + x1) / 2, row_y_bottom - 0.015, label,
            transform=trans, ha="center", va="top",
            fontproperties=fontprops, fontsize=fontsize, clip_on=False,
        )
        row += 1


def save_figure(fig, outfolder, filename_stub, exts=None, **savefig_kwargs):
    print(f"[TRACE] >>> Entering save_figure() - defined at line 2617 of {__file__}")
    """Single shared figure-writer for the whole pipeline (requirement 6):
    every plotting function in this script should call this instead of
    looping over extensions and building its own os.path.join calls, so
    the output layout stays identical everywhere.

    Instead of writing "<outfolder>/<stub>.png", "<outfolder>/<stub>.pdf",
    "<outfolder>/<stub>.svg" all mixed together in one folder (the
    original behaviour), figures are grouped by format into
    "<outfolder>/png/<stub>.png", "<outfolder>/pdf/<stub>.pdf",
    "<outfolder>/svg/<stub>.svg". The original naming convention for the
    stub itself (e.g. "plot_heatmap_pairwise_ari_simulations_<assembly>")
    is preserved unchanged -- only the directory the file lands in moves.

    Input:
        fig             -- the matplotlib Figure to save.
        outfolder       -- the run's top-level output folder (as passed
                            around everywhere else in the pipeline).
        filename_stub   -- file name without extension, e.g. the result
                            of "_".join([...]) at each call site.
        exts            -- iterable of extensions to write; defaults to
                            FIGURE_FORMATS (png, pdf, svg).
        **savefig_kwargs -- forwarded to fig.savefig (e.g. bbox_inches).
    Output: none (writes files to disk).
    """
    if exts is None:
        exts = FIGURE_FORMATS
    for ext in exts:
        ext_dir = os.path.join(outfolder, ext)
        os.makedirs(ext_dir, exist_ok=True)
        fig.savefig(
            os.path.join(ext_dir, f"{filename_stub}.{ext}"),
            **savefig_kwargs,
        )


def plot_nj_tree_from_matrix(
    mat, x, labels, namedict, outfolder, assembly, datatype, font_props,
    filename_prefix, title_suffix=None,
):
    print(f"[TRACE] >>> Entering plot_nj_tree_from_matrix() - defined at line 2653 of {__file__}")
    """Dendrogram companion for a symmetric method-vs-method similarity
    matrix (requirement 3): every symmetric heatmap this pipeline draws
    (pairwise ARI/AMI/purity/V-measure/F1) gets a matching dendrogram, so
    relationships between methods can be read off as a clustering
    hierarchy in addition to the heatmap's raw numbers.

    Method: `mat` holds a similarity score in ~[0, 1] (1 = identical
    clusterings) for every method pair, with only the lower triangle
    filled and the diagonal at 1.0 (see _plot_triangular_pairwise_heatmap).
    This is converted to a distance matrix via distance = 1 - similarity
    (clipped to >= 0 to absorb floating-point noise), mirrored into a
    full symmetric matrix, and then agglomerative hierarchical clustering
    (scipy's average-linkage/UPGMA `linkage`) is run on it to build a
    dendrogram, which is drawn with scipy's own plotting onto a
    matplotlib axis.

    Note: unlike the neighbour-joining tree this function used to
    produce, a dendrogram is a *rooted, ultrametric* hierarchy -- every
    tip ends at the same total height, and branch lengths reflect linkage
    (merge) distance rather than an estimate of independent evolutionary
    change along each branch. That's the correct/expected output for
    clustering method-similarity data (there's no meaningful "unrooted
    tree" interpretation for it), which is why NJ was swapped out here.

    Input:
        mat            -- same (n x n) matrix passed to
                           _plot_triangular_pairwise_heatmap (lower
                           triangle filled with a similarity score, upper
                           triangle NaN, diagonal 1.0).
        x              -- ordered list of combo keys (rows/columns of mat).
        labels         -- display labels for each row/column (same order
                           as x; used as the dendrogram's leaf labels).
        namedict, outfolder, assembly, datatype, font_props -- standard
                           plotting bookkeeping (see other plot functions).
        filename_prefix -- prefix used to build the output file name
                           (by convention, the heatmap's own
                           filename_prefix + "_dendrogram").
        title_suffix   -- optional short description of the metric shown
                           (e.g. the heatmap's cbar_label), used in the
                           figure title.
    Output: none (saves PNG/PDF/SVG figures to `outfolder` via
        save_figure). Silently skipped (with a warning) if fewer than 3
        methods are present, since a dendrogram isn't meaningful below
        that.
    """
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props

    n = len(x)
    if n < 3:
        warnings.warn(
            f"Fewer than 3 methods available for {assembly}/{datatype} "
            f"({filename_prefix}); skipping dendrogram",
            RuntimeWarning, stacklevel=2,
        )
        return

    # Mirror the lower triangle into a full symmetric similarity matrix,
    # then convert to a non-negative distance matrix.
    sim = np.array(mat, dtype=float)
    full_sim = np.where(np.isnan(sim), sim.T, sim)
    np.fill_diagonal(full_sim, 1.0)
    if np.isnan(full_sim).any():
        warnings.warn(
            f"Incomplete pairwise data for {assembly}/{datatype} "
            f"({filename_prefix}); skipping dendrogram",
            RuntimeWarning, stacklevel=2,
        )
        return
    dist = np.clip(1.0 - full_sim, 0.0, None)
    # A valid distance matrix needs an exactly-zero diagonal (it's
    # currently ~0 up to floating-point noise from the 1 - similarity
    # step above); squareform() requires this to be exact.
    np.fill_diagonal(dist, 0.0)

    # scipy's `linkage` wants a condensed distance matrix (the upper
    # triangle only, as a 1-D array) -- squareform() converts our square
    # symmetric matrix into that form.
    condensed_dist = squareform(dist, checks=False)
    Z = linkage(condensed_dist, method=DENDROGRAM_LINKAGE_METHOD)

    # --- readability tuning (requirement 3) -----------------------------
    # The default dendrogram layout packs leaves at a fixed ~1-unit
    # vertical spacing regardless of figure size, and labels sit flush
    # against the leaves, so with many/long method names (e.g.
    # "Connected components t=0.7* (NT) *") the leaf labels visually
    # collide with each other and with neighbouring branch lines. We fix
    # this by: (1) giving each leaf much more vertical room by scaling
    # figure height more aggressively with n, and (2) reserving extra
    # horizontal room for the (possibly long) leaf labels instead of
    # letting them run past the axes. None of this touches the
    # underlying clustering topology/merge distances.
    fig_h = max(5.0, n * 0.55 + 2.0)
    fig_w = max(9.0, n * 0.22 + 6.0)
    fig = plt.figure(1, dpi=150, figsize=(fig_w, fig_h))
    ax = fig.subplots()

    # orientation="left" puts the root at the left and leaves at the
    # right with the merge-distance axis running left-to-right, matching
    # the previous tree layout (branch length on the x-axis, tip labels
    # reading left-to-right on the right-hand side).
    dendrogram(
        Z, ax=ax, orientation="left", labels=list(labels),
        color_threshold=0, above_threshold_color="#4A4A4A",
    )

    # requirement 1: colour-code each leaf label by its method group
    # (sequence clustering/similarity, pangenome, sketching/embedding --
    # see METHOD_GROUP_NAMES/METHOD_GROUP_COLOURS) for consistent,
    # readable grouping across every dendrogram plot. `labels` is in the
    # same order as `x`; scipy's dendrogram() reorders the leaf tick
    # labels to match the clustering's leaf order, so match each tick
    # label back to its combo key by text rather than by position.
    label_to_combo = {lbl: combo for lbl, combo in zip(labels, x)}
    for text_obj in ax.get_yticklabels():
        text_obj.set_fontproperties(ibmplexsans)
        text_obj.set_fontsize(NJ_TREE_TIP_FONT_SIZE)
        combo = label_to_combo.get(text_obj.get_text())
        if combo is not None:
            group = get_method_group(combo)
            if group is not None:
                text_obj.set_color(METHOD_GROUP_COLOURS[group])
                text_obj.set_fontweight("bold")

    # Leave headroom to the right of the deepest merge so long leaf
    # labels have somewhere to sit instead of overlapping other branches.
    # Grab the ticks scipy chose (sensible distance values) before
    # extending the range, so the extra headroom doesn't grow spurious
    # ticks (e.g. negative "distance" values) of its own.
    # Note: with orientation="left" matplotlib's xlim is reversed (xmin is
    # the numerically larger, root-side value; xmax is 0, the leaf side),
    # so bound the kept ticks with min()/max() rather than assuming order.
    xmin, xmax = ax.get_xlim()
    xticks = ax.get_xticks()
    lo, hi = sorted((xmin, xmax))
    ax.set_xlim(xmin, xmin + (xmax - xmin) * 1.55)
    ax.set_xticks([t for t in xticks if lo <= t <= hi])

    ax.set_ylabel("")
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)

    ax.set_xlabel("Distance (1 − similarity)", fontproperties=ibmplexsans, fontsize=AXIS_TITLE_FONT_SIZE)
    for label in ax.get_xticklabels():
        label.set_fontproperties(ibmplexsans)

    datatype_suffix = get_datatype_title_suffix(datatype)
    title = f"Dendrogram — {namedict[assembly]}{datatype_suffix}"
    if title_suffix:
        title = f"Dendrogram, {title_suffix} — {namedict[assembly]}{datatype_suffix}"
    ax.set_title(title, fontproperties=ibmplexsansbold, fontsize=AXIS_TITLE_FONT_SIZE)

    # legend for the method-group colour coding, restricted to whichever
    # groups are actually present among this dendrogram's leaves
    groups_present = [g for g in METHOD_GROUP_COLOURS if any(get_method_group(c) == g for c in x)]
    if groups_present:
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color=METHOD_GROUP_COLOURS[g], lw=0, marker="s", markersize=8)
            for g in groups_present
        ]
        ax.legend(
            legend_handles, groups_present, loc="lower right",
            prop=ibmplexsans, fontsize=BASE_FONT_SIZE, frameon=True,
        )

    fig.tight_layout()
    save_figure(fig, outfolder, "_".join([filename_prefix, datatype, assembly]), bbox_inches="tight")
    plt.close(fig)


def plotter(theargs):
    print(f"[TRACE] >>> Entering plotter() - defined at line 2751 of {__file__}")
    """Line plot ("c-plot"): shows how a given clustering-quality or
    performance metric varies with the minimum sequence-identity threshold
    `c` used by each clustering method, for one simulated assembly.

    Figure interpretation:
        - X axis: minimum sequence identity threshold `c` (adimensional,
          0-1) that each clusterer was run with -- i.e. how similar two
          sequences must be to be merged into the same gene cluster.
        - Y axis: the requested metric `name` (e.g. Adjusted Rand index,
          purity, adjusted mutual information, V-measure, or runtime in
          seconds), averaged over all available random-seed replicates at
          that (clusterer, seqtype, c) combination.
        - Each coloured line is one (clusterer, sequence-type) combination
          (e.g. "CD-HIT (NT)"), using the fixed CONFIGDICT_COLOURS palette
          so the same method always gets the same colour across all plots
          in this script; markers are individual data points, and the
          shaded band around each line is +/- 1 standard deviation across
          seeds (only drawn where at least 2 seeds are available).
        - Sketch/HDBSCAN and embeddings/HDBSCAN methods are excluded here
          since they are run once per seed on a fixed embedding and have
          no `c` parameter to sweep (see FAMILY_METHOD_NAMES).
    Biological/methodological reading: for the agreement metrics (ARI,
    purity, AMI, V-measure), a curve that stays high and flat across `c`
    indicates a clusterer robust to the identity-threshold choice, while a
    curve that degrades sharply at high `c` suggests the method starts
    fragmenting true gene families once sequences must be near-identical
    to cluster together. For "runtime", the plot instead shows raw
    computational cost as a function of `c` (Y axis switched to log scale
    below, except for runtime).

    Input:  theargs -- (name, datadf, namedict, outfolder, assembly,
        datatype, font_props) tuple; `name` selects which metric column
        of `datadf` to plot, `datadf` is the full simulation results
        table (outdf), and the rest control labelling/output location.
    Output: none (saves a PNG/PDF/SVG figure to `outfolder`). When `name`
        is "adj_rand_index" or "adj_mutual_info", the companion CSV
        (written via write_metric_csv) also includes a "mean_pvalue"
        column -- the mean empirical permutation p-value (see
        permutation_test_agreement) across seed replicates at that
        (method, c) point -- alongside the plotted mean/std/n; for every
        other metric that column is NaN (no p-value is computed).
    """
    name, datadf, namedict, outfolder, assembly, datatype, font_props = theargs
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props
    print(f"\t- Plotting c-plot {name} for simulations of {namedict[assembly]}")
    subdf = datadf[
        (datadf.index.get_level_values("assembly") == assembly)
        & (datadf.index.get_level_values("simulations") == (datatype == "simulations"))
    ]

    if subdf.empty:
        warnings.warn(
            f"No rows available for {assembly}/{datatype}/{name}; skipping plot",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    fig = plt.figure(1, dpi=150, figsize=DEFAULT_FIGSIZE)
    ax = fig.subplots()
    xs = list(set(list(subdf["c"].astype(float))))
    xs.sort()
    clusterers = list(set(list(subdf.index.get_level_values("clusterer"))))

    # p-value column to accompany `name`, when one exists (ARI/AMI only --
    # see calculate_values_from_cluster_matrix). None for every other metric.
    pvalue_col = {
        "adj_rand_index": ADJ_RAND_INDEX_PVALUE_COL,
        "adj_mutual_info": ADJ_MUTUAL_INFO_PVALUE_COL,
    }.get(name)

    csv_rows = []  # requirement 6: exact mean/std/n behind every point on this c-sweep plot

    for seqtype in SEQTYPES:
        availcl = list(set(list(subdf[subdf["st"] == seqtype].index.get_level_values("clusterer"))))
        ynams = [
            el + "/" + seqtype
            for el in clusterers
            if el in availcl
            and el not in SKETCH_METHOD_NAMES  # no c sweep for sketch/HDBSCAN methods
            and el not in EMBED_METHOD_NAMES  # no c sweep for embeddings/HDBSCAN methods
            and not (name == "runtime" and el == "panx")  # panx is a runtime outlier
            and (el + "/" + seqtype) in FANCYDICT
            and (el + "/" + seqtype) in CONFIGDICT_COLOURS
        ]
        ymean = np.zeros((len(ynams), len(xs)))
        ystd = np.zeros((len(ynams), len(xs)))
        ycount = np.zeros((len(ynams), len(xs)), dtype=int)

        for cluster_index, ynam in enumerate(ynams):
            for x_index, x_value in enumerate(xs):
                tmpdf = subdf[
                    (subdf["st"] == seqtype)
                    & (subdf.index.get_level_values("simulations") == (datatype == "simulations"))
                    & (subdf.index.get_level_values("assembly") == assembly)
                    & (subdf.index.get_level_values("clusterer") == ynam.split("/")[0])
                    & (subdf["c"] == x_value)
                ][name].astype(float)
                ymean[cluster_index, x_index] = tmpdf.mean()
                ycount[cluster_index, x_index] = tmpdf.count()
                ystd[cluster_index, x_index] = tmpdf.std() if tmpdf.count() >= 2 else 0.0
                # mean permutation p-value at this (method, c) point, when
                # `name` is ARI or AMI; NaN (no p-value) for every other metric.
                mean_pvalue = np.nan
                if pvalue_col is not None:
                    pdf = subdf[
                        (subdf["st"] == seqtype)
                        & (subdf.index.get_level_values("simulations") == (datatype == "simulations"))
                        & (subdf.index.get_level_values("assembly") == assembly)
                        & (subdf.index.get_level_values("clusterer") == ynam.split("/")[0])
                        & (subdf["c"] == x_value)
                    ][pvalue_col].astype(float)
                    mean_pvalue = pdf.mean()
                csv_rows.append({
                    "method": FANCYDICT[ynam],
                    "combo": ynam,
                    "metric": name,
                    "c": x_value,
                    "mean": ymean[cluster_index, x_index],
                    "std": ystd[cluster_index, x_index],
                    "n": ycount[cluster_index, x_index],
                    "mean_pvalue": mean_pvalue,
                })

        for y_index, ynam in enumerate(ynams):
            ax.plot(xs, ymean[y_index, :], "-o", c=CONFIGDICT_COLOURS[ynam], label=FANCYDICT[ynam])
            if np.any(ycount[y_index, :] >= 2):
                ax.fill_between(
                    xs,
                    [
                        max(ymean[y_index, j] - ystd[y_index, j], 0.0)
                        for j in range(len(ymean[y_index, :]))
                    ],
                    ymean[y_index, :] + ystd[y_index, :],
                    where=ycount[y_index, :] >= 2,
                    alpha=0.5,
                    edgecolor=CONFIGDICT_COLOURS[ynam],
                    facecolor=CONFIGDICT_COLOURS[ynam],
                )

    if name in CONFIGDICT and "ylimits_c" in CONFIGDICT[name]:
        ax.set_ylim(CONFIGDICT[name]["ylimits_c"][0], CONFIGDICT[name]["ylimits_c"][1])
    elif name in CONFIGDICT and "ylimits" in CONFIGDICT[name]:
        ax.set_ylim(CONFIGDICT[name]["ylimits"][0], CONFIGDICT[name]["ylimits"][1])
    else:
        ax.set_ylim(0.0, None)

    ax.set_xlabel("minimum sequence identity (adim.)", fontproperties=ibmplexsans, loc="right", fontsize=AXIS_TITLE_FONT_SIZE)
    ax.set_ylabel(
        CONFIGDICT[name]["ylabel"] if name in CONFIGDICT and "ylabel" in CONFIGDICT[name] else name,
        fontproperties=ibmplexsans,
        loc="top",
        fontsize=AXIS_TITLE_FONT_SIZE,
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="major", direction="in")
    ax.tick_params(which="minor", direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.ticklabel_format(useMathText=True)
    ax.get_yaxis().get_offset_text().set_x(-0.075)
    ax.get_yaxis().get_offset_text().set_fontproperties(ibmplexsans)
    ax.set_xlim(xs[0], xs[-1])
    
    outnamescaff = name.replace(" ", "").replace("#", "NumberOf")

    write_metric_csv(pd.DataFrame(csv_rows), outfolder, "_".join(["plot_c", datatype, assembly, outnamescaff]))

    if outnamescaff!= "runtime" :
        # change to log scale
        ax.set_yscale("log")
        #ax.set_ylim(0.97, 1.001)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)

    plt.text(0, 1.01, namedict[assembly], fontproperties=ibmplexsansitalics, horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)
    plt.legend(loc="best", frameon=False, prop=ibmplexsans, handlelength=0.5, handletextpad=0.75, labelspacing=0.3)

    save_figure(fig, outfolder, "_".join(["plot_c", datatype, assembly, outnamescaff]), bbox_inches="tight")
    fig.clf()
    del fig, ax

# here we fix a c in comparaison to the previous plots
# now we compare clustering methods between them for this fixed c

def _find_runtime_axis_break(values):
    print(f"[TRACE] >>> Entering _find_runtime_axis_break() - defined near line 3100 of {__file__}")
    """Look for a natural gap in a set of runtime values that's large
    enough to justify a broken y-axis (requirement 5): fast methods
    (~seconds) and slow methods (~thousands of seconds) otherwise get
    squashed onto the same linear axis, making the fast methods
    indistinguishable from zero.

    Method: sort the positive values, and find the pair of consecutive
    values with the largest ratio between them. If that ratio is at
    least 5x, treat it as the break point; otherwise there's no
    meaningful bimodal split and no break is needed.

    Input:  values -- iterable of runtime means (may include NaN/0).
    Output: (low_max, high_min) tuple marking the top of the "low" axis
            and the bottom of the "high" axis, or None if no break is
            warranted (fewer than 2 positive values, or no gap >= 5x).
    """
    positive_sorted = sorted(v for v in values if v is not None and not np.isnan(v) and v > 0)
    if len(positive_sorted) < 2:
        return None
    ratios = [positive_sorted[i + 1] / positive_sorted[i] for i in range(len(positive_sorted) - 1)]
    best_idx = int(np.argmax(ratios))
    if ratios[best_idx] < 5.0:
        return None
    return positive_sorted[best_idx], positive_sorted[best_idx + 1]


def _plot_runtime_broken_axis(
    x, x_fancy, ymean, ystd, ycount, namedict, outfolder, assembly, datatype,
    font_props,
):
    print(f"[TRACE] >>> Entering _plot_runtime_broken_axis() - defined near line 3100 of {__file__}")
    """Runtime bar plot with a broken y-axis (requirement 5): when
    runtimes span multiple orders of magnitude (e.g. ~3 s for CD-HIT vs
    ~7000 s for sketching methods), a single linear axis makes the fast
    methods invisible. This draws the same bars on two stacked axes --
    a short "high" axis on top covering the slow methods' range, and a
    taller "low" axis on the bottom covering the fast methods' range --
    with a diagonal break mark where the axis is cut, so every method's
    bar height is readable regardless of scale, while still keeping all
    methods on one figure for direct comparison.

    Falls back to a single ordinary axis (via the shared bar-drawing
    logic below) if no runtime values span a big enough gap to need a
    break (see _find_runtime_axis_break).

    Input/Output: same data as plotter_pointplots's runtime branch;
        saves the figure via save_figure exactly like the non-runtime
        path, under the same filename stub ("plot_point_<datatype>_
        <assembly>_runtime").
    """
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props
    positions = list(range(len(x)))
    bar_width = 0.5
    break_pts = _find_runtime_axis_break(ymean)

    def _draw_bars(ax):
        for index in positions:
            ax.bar(
                index, ymean[index], bar_width,
                color=CONFIGDICT_COLOURS[x[index]], label=x_fancy[index],
            )
            if ycount[index] >= 2:
                lower_err = min(ystd[index], ymean[index])
                ax.errorbar(
                    index, ymean[index],
                    yerr=[[max(0, lower_err)], [ystd[index]]],
                    fmt="none", color="black", capsize=4.0, linewidth=1.0,
                )

    if break_pts is None:
        # no large enough gap -- draw as a single ordinary axis
        fig = plt.figure(1, dpi=150, figsize=DEFAULT_FIGSIZE)
        ax = fig.subplots()
        _draw_bars(ax)
        ax.set_ylim(0.0, None)
        ax.set_xticks(positions)
        ax.set_xticklabels(x_fancy, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xlabel("Clusterer", fontproperties=ibmplexsans, loc="right", fontsize=AXIS_TITLE_FONT_SIZE)
        ax.set_ylabel(CONFIGDICT["runtime"]["ylabel"], fontproperties=ibmplexsans, loc="top", fontsize=AXIS_TITLE_FONT_SIZE)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="major", direction="in")
        ax.tick_params(which="minor", direction="in")
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(ibmplexsans)
        plt.text(0, 1.01, namedict[assembly], fontproperties=ibmplexsansitalics, horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)
        add_sketch_bracket(ax, x, positions, bar_width=bar_width, fontprops=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1)
        family_footnote = get_family_footnote(x)
        if family_footnote is not None:
            plt.text(
                0.5, -0.44, family_footnote,
                fontproperties=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1,
                horizontalalignment="center", verticalalignment="top", transform=ax.transAxes,
            )
        save_figure(fig, outfolder, "_".join(["plot_point", datatype, assembly, "runtime"]), bbox_inches="tight")
        fig.clf()
        del fig, ax
        return

    low_max, high_min = break_pts
    top_max = max(v for v in ymean if v is not None and not np.isnan(v))
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, dpi=150,
        figsize=(DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1] + 1.0),
        gridspec_kw={"height_ratios": [1, 1.6], "hspace": 0.08},
    )
    _draw_bars(ax_top)
    _draw_bars(ax_bot)

    ax_top.set_ylim(high_min * 0.85, top_max * 1.08)
    ax_bot.set_ylim(0.0, low_max * 1.35)

    # hide the spine/ticks between the two axes and draw diagonal break marks
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, top=False, bottom=False)
    ax_bot.xaxis.tick_bottom()

    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - 0.7 * d, 1 + 0.7 * d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - 0.7 * d, 1 + 0.7 * d), **kwargs)

    ax_bot.set_xticks(positions)
    ax_bot.set_xticklabels(x_fancy, rotation=35, ha="right", rotation_mode="anchor")
    ax_bot.set_xlabel("Clusterer", fontproperties=ibmplexsans, loc="right", fontsize=AXIS_TITLE_FONT_SIZE)
    fig.text(
        0.04, 0.5, CONFIGDICT["runtime"]["ylabel"], va="center", rotation="vertical",
        fontproperties=ibmplexsans, fontsize=AXIS_TITLE_FONT_SIZE,
    )

    for ax in (ax_top, ax_bot):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="major", direction="in")
        ax.tick_params(which="minor", direction="in")
        ax.yaxis.set_ticks_position("both")
        for label in ax.get_yticklabels():
            label.set_fontproperties(ibmplexsans)
    for label in ax_bot.get_xticklabels():
        label.set_fontproperties(ibmplexsans)

    ax_top.text(0, 1.02, namedict[assembly], fontproperties=ibmplexsansitalics, horizontalalignment="left", verticalalignment="bottom", transform=ax_top.transAxes)

    add_sketch_bracket(ax_bot, x, positions, bar_width=bar_width, fontprops=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1)

    family_footnote = get_family_footnote(x)
    if family_footnote is not None:
        ax_bot.text(
            0.5, -0.5, family_footnote,
            fontproperties=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1,
            horizontalalignment="center", verticalalignment="top", transform=ax_bot.transAxes,
        )

    save_figure(fig, outfolder, "_".join(["plot_point", datatype, assembly, "runtime"]), bbox_inches="tight")
    plt.close(fig)


def plotter_pointplots(theargs):
    print(f"[TRACE] >>> Entering plotter_pointplots() - defined at line 2899 of {__file__}")
    """Bar/point plot comparing all clustering methods to each other at a
    FIXED sequence-identity threshold (`c` == DEFAULT_PARAMS["c"], the
    "default" operating point), for one simulated assembly and one metric.

    Figure interpretation:
        - X axis: one bar per (clusterer, sequence-type) combination
          (e.g. "CD-HIT (NT)", "Panaroo"), in COMBO_ORDER, coloured by
          CONFIGDICT_COLOURS (the same palette used everywhere else in
          this script, so a given method is always the same colour).
        - Y axis: the requested metric `name` (e.g. Adjusted Rand index,
          purity, runtime), averaged across all available random-seed
          replicates for that method at the default `c`.
        - Black error bars show +/- 1 standard deviation across seeds
          (only drawn when >= 2 seeds are available); for "runtime" the
          lower whisker is clipped so it can't extend below zero seconds.
        - A bracket annotation under the sketch/embeddings bars marks
          them as a distinct method "family" (see add_sketch_bracket),
          since those methods don't have a `c` parameter at all.
    Biological/methodological reading: taller bars for the agreement
    metrics mean that method recovers the simulated ground-truth gene
    families more faithfully at the field-standard identity threshold;
    for "runtime" taller means slower. This plot answers "which method
    performs best in practice, at the threshold people actually use?",
    complementing plotter's "how does performance change as the
    threshold is varied?".

    Input/Output: same shape as plotter (see its docstring); this
        function additionally filters out DIAMOND/panx-runtime outliers
        from the runtime plot and orders methods via build_ordered_combo_list.
        For "adj_rand_index"/"adj_mutual_info", bars are additionally
        marked with "†" above the error bar when that method's mean
        permutation p-value (see permutation_test_agreement) is below
        SIGNIFICANCE_ALPHA, and the companion CSV gets a "mean_pvalue"
        column (NaN for every other metric).
    """
    name, datadf, namedict, outfolder, assembly, datatype, font_props = theargs
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props
    print(f"\t- Plotting point-plot {name} for simulations of {namedict[assembly]}")
    subdf = datadf[
        (datadf.index.get_level_values("assembly") == assembly)
        & (datadf.index.get_level_values("simulations") == (datatype == "simulations"))
        & (datadf["c"] == DEFAULT_PARAMS["c"])
    ]
    if subdf.empty:
        warnings.warn(
            f"No rows available for {assembly}/{datatype}/{name}; skipping point plot",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    clusterers = list(set(list(subdf.index.get_level_values("clusterer"))))
    x = build_ordered_combo_list(subdf, name)
    if name == "runtime":
        x = [combo for combo in x if combo.split("/")[0] != "panx"]  # panx is a runtime outlier

    if not x:
        warnings.warn(
            f"No clusterer/sequence-type rows available for {assembly}/{datatype}/{name}; skipping point plot",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    # p-value column to accompany `name`, when one exists (ARI/AMI only --
    # see calculate_values_from_cluster_matrix). None for every other metric.
    pvalue_col = {
        "adj_rand_index": ADJ_RAND_INDEX_PVALUE_COL,
        "adj_mutual_info": ADJ_MUTUAL_INFO_PVALUE_COL,
    }.get(name)

    x_fancy = [FANCYDICT[value] for value in x]
    ymean = []
    ystd = []
    ycount = []
    ypvalue = []  # mean permutation p-value per method, only populated when pvalue_col is set
    for x_value in x:
        tmpdf = subdf[
            (subdf.index.get_level_values("simulations") == (datatype == "simulations"))
            & (subdf.index.get_level_values("assembly") == assembly)
            & (subdf.index.get_level_values("clusterer") == x_value.split("/")[0])
            & (subdf["st"] == x_value.split("/")[1])
        ]
        vals = tmpdf[name].astype(float)
        ymean.append(vals.mean())
        ycount.append(vals.count())
        ystd.append(vals.std() if vals.count() >= 2 else 0.0)
        if pvalue_col is not None:
            ypvalue.append(tmpdf[pvalue_col].astype(float).mean())
        else:
            ypvalue.append(np.nan)

    outnamescaff = name.replace(" ", "").replace("#", "NumberOf")

    # requirement 6: export the exact (unrounded) mean/std/n values behind
    # this bar plot to CSV -- same x/ymean/ystd/ycount arrays used to draw
    # the figure below, so there's no risk of the CSV and plot disagreeing.
    # For ARI/AMI, also export the mean permutation-test p-value per method
    # (mean_pvalue is NaN for every other metric, since no p-value exists).
    write_metric_csv(
        pd.DataFrame({
            "method": x_fancy,
            "combo": x,
            "metric": name,
            "mean": ymean,
            "std": ystd,
            "n": ycount,
            "mean_pvalue": ypvalue,
        }),
        outfolder, "_".join(["plot_point", datatype, assembly, outnamescaff]),
    )

    if name == "runtime":
        _plot_runtime_broken_axis(
            x, x_fancy, ymean, ystd, ycount, namedict, outfolder, assembly,
            datatype, font_props,
        )
        return

    fig = plt.figure(1, dpi=150, figsize=DEFAULT_FIGSIZE)
    ax = fig.subplots()
    positions = list(range(len(x)))
    bar_width = 0.5

    for index in positions:
        ax.bar(
            index,
            ymean[index],
            bar_width,
            color=CONFIGDICT_COLOURS[x[index]],
            label=x_fancy[index],
        )
        if ycount[index] >= 2:
            if outnamescaff == "runtime":
                lower_err = min(ystd[index], ymean[index])
                ax.errorbar(
                    index,
                    ymean[index],
                    yerr=[[max(0, lower_err)], [ystd[index]]],
                    fmt="none",
                    color="black",
                    capsize=4.0,
                    linewidth=1.0,
                ) 
            else :
                
                ax.errorbar(
                    index,
                    ymean[index],
                    yerr=[[ystd[index] if ymean[index] > ystd[index] else ymean[index]], [ystd[index]]],
                    fmt="none",
                    color="black",
                    capsize=4.0,
                    linewidth=1.0,
                )
        # significance marker for ARI/AMI (see pvalue_col above): "†" above
        # the bar/error-bar when the mean permutation p-value is below
        # SIGNIFICANCE_ALPHA, i.e. this method's agreement with the ground
        # truth is unlikely to be chance.
        if pvalue_col is not None and not np.isnan(ypvalue[index]) and ypvalue[index] < SIGNIFICANCE_ALPHA:
            marker_y = ymean[index] + (ystd[index] if ycount[index] >= 2 else 0.0)
            ax.text(
                index, marker_y, "†",
                ha="center", va="bottom", fontsize=BASE_FONT_SIZE + 1,
                fontproperties=ibmplexsans,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(x_fancy, rotation=35, ha="right", rotation_mode="anchor")

    if name in CONFIGDICT and "ylimits" in CONFIGDICT[name]:
        ax.set_ylim(CONFIGDICT[name]["ylimits"][0], CONFIGDICT[name]["ylimits"][1])
    else:
        ax.set_ylim(0.0, None)

    ax.set_xlabel("Clusterer", fontproperties=ibmplexsans, loc="right", fontsize=AXIS_TITLE_FONT_SIZE)
    ax.set_ylabel(
        CONFIGDICT[name]["ylabel"] if name in CONFIGDICT and "ylabel" in CONFIGDICT[name] else name,
        fontproperties=ibmplexsans,
        loc="top",
        fontsize=AXIS_TITLE_FONT_SIZE,
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="major", direction="in")
    ax.tick_params(which="minor", direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.get_yaxis().get_offset_text().set_x(-0.075)
    ax.get_yaxis().get_offset_text().set_fontproperties(ibmplexsans)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)

    plt.text(0, 1.01, namedict[assembly], fontproperties=ibmplexsansitalics, horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)

    # bracket under the sketching methods, so readers see they're one family
    add_sketch_bracket(ax, x, positions, bar_width=bar_width, fontprops=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1)

    footnote_lines = []
    family_footnote = get_family_footnote(x)
    if family_footnote is not None:
        footnote_lines.append(family_footnote)
    if pvalue_col is not None and any(not np.isnan(p) and p < SIGNIFICANCE_ALPHA for p in ypvalue):
        footnote_lines.append(
            f"† mean permutation-test p < {SIGNIFICANCE_ALPHA:g} vs ground truth "
            "(see permutation_test_agreement)"
        )
    if footnote_lines:
        plt.text(
            0.5, -0.44, "\n".join(footnote_lines),
            fontproperties=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1,
            horizontalalignment="center", verticalalignment="top", transform=ax.transAxes,
        )

    save_figure(fig, outfolder, "_".join(["plot_point", datatype, assembly, outnamescaff]), bbox_inches="tight")
    fig.clf()
    del fig, ax

def number_of_clusters_violin(theargs):
    print(f"[TRACE] >>> Entering number_of_clusters_violin() - defined at line 3057 of {__file__}")
    """Violin plot of the number of clusters ("n_clusters", i.e. inferred
    gene families) each method produces at the default sequence-identity
    threshold, showing the full distribution across random-seed replicates
    rather than just its mean (contrast with plotter_pointplots's bar+
    error-bar summary of the same underlying data).

    Figure interpretation:
        - X axis: one violin per (clusterer, sequence-type) combination.
        - Y axis: number of clusters inferred by that method (count of
          distinct gene families/clusters it output).
        - Violin width at a given Y value is proportional to how many
          seed replicates produced that cluster count -- a wide, short
          violin means the method is very consistent across seeds; a
          tall, thin violin means the cluster count varies a lot from
          one random simulation replicate to the next.
    Biological reading: the simulated ground truth has a known,
    fixed number of true gene families, so a method whose violin sits
    close to that number (and is narrow) is both accurate and
    reproducible; systematic over- or under-clustering shows up as the
    whole violin sitting well above or below the true value.

    Input/Output: same shape/behaviour as plotter_pointplots, except that
        theargs carries one extra trailing element, gt_stats: for
        simulation data's "n_clusters" plot this is the dict returned by
        get_simulation_ground_truth_cluster_stats() (mean/min/max number
        of true clusters across seeds, reusing the same
        fast_count_groundtruth_clusters.sh logic); it is None for every
        other case (real data, or gt_stats unavailable), in which case
        no overlay is drawn and the plot is unchanged from before.
    """
    name, datadf, namedict, outfolder, assembly, datatype, font_props, gt_stats = theargs
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props

    print(f"\t- Plotting violin {name} for simulations of {namedict[assembly]}")
    subdf = datadf[
     
       (datadf.index.get_level_values("assembly") == assembly)
        & (datadf.index.get_level_values("simulations") == (datatype == "simulations"))
        & (datadf["c"] == DEFAULT_PARAMS["c"])
    ]

    if subdf.empty:
        warnings.warn(
            f"No rows available for {assembly}/{datatype}/{name}; skipping violin plot",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    clusterers = list(set(list(subdf.index.get_level_values("clusterer"))))
    x = build_ordered_combo_list(subdf, name)

    if not x:
        warnings.warn(
            f"No clusterer/sequence-type rows available for {assembly}/{datatype}/{name}; skipping violin plot",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    x_fancy = [FANCYDICT[value] for value in x]
    data, counts = [], []
    for x_value in x:
        tmpdf = subdf[
            (subdf.index.get_level_values("simulations") == (datatype == "simulations"))
            & (subdf.index.get_level_values("assembly") == assembly)
            & (subdf.index.get_level_values("clusterer") == x_value.split("/")[0])
            & (subdf["st"] == x_value.split("/")[1])
        ][name].astype(float)
        data.append(tmpdf.values)
        counts.append(tmpdf.count())

    # requirement 6: raw per-seed values behind every violin, exact precision
    write_metric_csv(
        pd.DataFrame([
            {"method": FANCYDICT[x_value], "combo": x_value, "metric": name, "seed_index": i, "value": v}
            for x_value, values in zip(x, data)
            for i, v in enumerate(values)
        ]),
        outfolder, "_".join(["plot_violin", datatype, assembly, name.replace(" ", "").replace("#", "NumberOf")]),
    )

    fig = plt.figure(1, dpi=150, figsize=DEFAULT_FIGSIZE)
    ax = fig.subplots()
    positions = list(range(1, len(x) + 1))

    # violinplot needs >=2 points per group to estimate a density;
    # groups with a single seed get plotted as a lone point instead
    violin_positions = [p for p, c in zip(positions, counts) if c >= 2]
    violin_data = [d for d, c in zip(data, counts) if c >= 2]
    violin_keys = [k for k, c in zip(x, counts) if c >= 2]

    if violin_data:
        parts = ax.violinplot(violin_data, positions=violin_positions, showmeans=True, showextrema=True)
        for body, key in zip(parts["bodies"], violin_keys):
            body.set_facecolor(CONFIGDICT_COLOURS[key])
            body.set_edgecolor(CONFIGDICT_COLOURS[key])
            body.set_alpha(0.6)
        for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
            if partname in parts:
                parts[partname].set_edgecolor("black")
                parts[partname].set_linewidth(0.8)

    for p, d, c, key in zip(positions, data, counts, x):
        if c < 2 and len(d):
            ax.plot(p, d[0], "o", c=CONFIGDICT_COLOURS[key])

    # Ground-truth number-of-clusters overlay (simulation data only, and
    # only on the "n_clusters" plot -- gt_stats is None otherwise, e.g.
    # for real data, or if no truth_matrix files could be located). Kept
    # deliberately subtle (thin lines, low alpha, drawn behind the
    # violins) so it doesn't obscure the per-method distributions: a
    # very thin shaded band spans min-to-max across the whole plot
    # width, with a slightly bolder thin line marking the mean.
    if datatype == "simulations" and name == "n_clusters" and gt_stats is not None:
        gt_colour = "dimgray"
        ax.axhspan(
            gt_stats["min"], gt_stats["max"],
            color=gt_colour, alpha=0.10, linewidth=0, zorder=0.5,
        )
        ax.axhline(
            gt_stats["mean"], color=gt_colour, linestyle="-",
            linewidth=0.8, alpha=0.75, zorder=0.5,
        )
        for bound in ("min", "max"):
            ax.axhline(
                gt_stats[bound], color=gt_colour, linestyle=":",
                linewidth=0.6, alpha=0.6, zorder=0.5,
            )
        ax.text(
            1.0, gt_stats["mean"],
            f" true n_clusters: mean={gt_stats['mean']:.0f}, "
            f"min={gt_stats['min']:.0f}, max={gt_stats['max']:.0f} "
            f"(n={gt_stats['n_seeds']} seeds)",
            transform=ax.get_yaxis_transform(),
            fontproperties=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 2,
            color=gt_colour, va="center", ha="left",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(x_fancy, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylim(100, 3000)

    ax.set_xlabel("Clusterer", fontproperties=ibmplexsans, loc="right", fontsize=AXIS_TITLE_FONT_SIZE)
    ax.set_ylabel("Number of clusters (adim.)", fontproperties=ibmplexsans, loc="top", fontsize=AXIS_TITLE_FONT_SIZE)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="major", direction="in")
    ax.tick_params(which="minor", direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.get_yaxis().get_offset_text().set_x(-0.075)
    ax.get_yaxis().get_offset_text().set_fontproperties(ibmplexsans)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)

    plt.text(0, 1.01, namedict[assembly], fontproperties=ibmplexsansitalics, horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)

    # bracket under the sketching methods, so readers see they're one family
    add_sketch_bracket(ax, x, positions, bar_width=0.5, fontprops=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1)

    family_footnote = get_family_footnote(x)
    if family_footnote is not None:
        plt.text(
            0.5, -0.44, family_footnote,
            fontproperties=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1,
            horizontalalignment="center", verticalalignment="top", transform=ax.transAxes,
        )

    outnamescaff = name.replace(" ", "").replace("#", "NumberOf")
    save_figure(fig, outfolder, "_".join(["plot_violin", datatype, assembly, outnamescaff]), bbox_inches="tight")
    fig.clf()
    del fig, ax


def number_of_clusters_stacked_bar(theargs):
    print(f"[TRACE] >>> Entering number_of_clusters_stacked_bar() - defined at line 3184 of {__file__}")
    """Stacked bar chart breaking down each method's total cluster count
    into singleton clusters, pair (2-member) clusters, and everything
    else, at the default sequence-identity threshold.

    Figure interpretation:
        - X axis: one stacked bar per (clusterer, sequence-type)
          combination.
        - Y axis: number of clusters (count).
        - Each bar is split into (typically) three stacked segments:
          singleton clusters (n_clusters with exactly 1 gene, see
          count_singleton_clusters), pair clusters (exactly 2 genes, see
          count_pairs_clusters), and the remainder (clusters with 3+
          genes) -- segment colours/labels are set where the bars are
          actually drawn further down in this function.
    Biological reading: singleton-heavy bars suggest a method is
    reporting a lot of strain-specific/orphan genes (or over-splitting
    true families into fragments), whereas a bar dominated by the "3+"
    segment suggests larger, well-populated gene families are being
    correctly merged. Comparing the relative segment sizes across
    methods highlights differences in clustering "granularity" that a
    single total cluster-count number would hide.

    Input/Output: same shape/behaviour as plotter_pointplots.
    """
    name, datadf, namedict, outfolder, assembly, datatype, font_props = theargs
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props

    print(f"\t- Plotting stacked bar {name} for simulations of {namedict[assembly]}")
    subdf = datadf[
        (datadf.index.get_level_values("assembly") == assembly)
        & (datadf.index.get_level_values("simulations") == (datatype == "simulations"))
        & (datadf["c"] == DEFAULT_PARAMS["c"])
    ]

    if subdf.empty:
        warnings.warn(
            f"No rows available for {assembly}/{datatype}/{name}; skipping stacked bar",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    x = []
    for combo in COMBO_ORDER:
        clusterer, seqtype = combo.split("/")
        if combo not in FANCYDICT or combo not in CONFIGDICT_COLOURS:
            continue
        if len(list(subdf[
            (subdf.index.get_level_values("clusterer") == clusterer)
            & (subdf["st"] == seqtype)
        ]["n_clusters"])):
            x.append(combo)

    if not x:
        warnings.warn(
            f"No clusterer/sequence-type rows available for {assembly}/{datatype}; skipping stacked bar",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    x_fancy = [FANCYDICT[value] for value in x]
    mean_total, mean_singletons, mean_pairs = [], [], []

    for x_value in x:
        tmpdf = subdf[
            (subdf.index.get_level_values("simulations") == (datatype == "simulations"))
            & (subdf.index.get_level_values("assembly") == assembly)
            & (subdf.index.get_level_values("clusterer") == x_value.split("/")[0])
            & (subdf["st"] == x_value.split("/")[1])
        ]
        mean_total.append(tmpdf["n_clusters"].astype(float).mean())
        mean_singletons.append(tmpdf["n_singletons"].astype(float).mean())
        mean_pairs.append(tmpdf["n_pairs"].astype(float).mean())

    mean_rest = [t - s - p for t, s, p in zip(mean_total, mean_singletons, mean_pairs)]

    # requirement 6: exact singleton/pair/rest/total counts behind this bar
    write_metric_csv(
        pd.DataFrame({
            "method": x_fancy,
            "combo": x,
            "mean_total_clusters": mean_total,
            "mean_singleton_clusters": mean_singletons,
            "mean_pair_clusters": mean_pairs,
            "mean_other_clusters": mean_rest,
        }),
        outfolder, "_".join(["plot_stackedbar", datatype, assembly, "n_clusters"]),
    )

    positions = list(range(len(x)))
    bar_width = 0.5

    fig = plt.figure(1, dpi=150, figsize=(max(DEFAULT_FIGSIZE[0], len(x) * 1.6), DEFAULT_FIGSIZE[1]))
    ax = fig.subplots()

    import matplotlib.patches as mpatches
    for i, x_value in enumerate(x):
        col            = CONFIGDICT_COLOURS[x_value]   # darkest  → other clusters (bottom)
        col_pairs      = col + "BB"                    # mid      → pairs
        col_singletons = col + "66"                    # lightest → singletons (top)

        # bottom segment: other clusters (darkest)
        ax.bar(positions[i], mean_rest[i], bar_width, color=col)
        # middle segment: pairs
        ax.bar(positions[i], mean_pairs[i], bar_width,
               bottom=mean_rest[i], color=col_pairs)
        # top segment: singletons (lightest)
        ax.bar(positions[i], mean_singletons[i], bar_width,
               bottom=mean_rest[i] + mean_pairs[i], color=col_singletons)

        # total count above bar
        ax.text(
            positions[i], mean_total[i] * 1.03,
            f"{int(round(mean_total[i]))}",
            ha="center", va="bottom",
            fontproperties=ibmplexsans,
            fontsize=9,
        )

    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0, np.nanmax(mean_total) * 1.2)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_fancy, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Clusterer", fontproperties=ibmplexsans, loc="right", fontsize=AXIS_TITLE_FONT_SIZE)
    ax.set_ylabel("Number of clusters (adim.)", fontproperties=ibmplexsans, loc="top", fontsize=AXIS_TITLE_FONT_SIZE)

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="major", direction="in")
    ax.tick_params(which="minor", direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.get_yaxis().get_offset_text().set_x(-0.075)
    ax.get_yaxis().get_offset_text().set_fontproperties(ibmplexsans)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)

    plt.text(0, 1.01, namedict[assembly], fontproperties=ibmplexsansitalics,
             horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)

    # bracket(s) under each method family (sketch, embeddings, ...), so
    # readers see at a glance that they're one family of methods.
    add_sketch_bracket(
        ax, x, positions, bar_width=bar_width, y_top=-0.30, y_bottom=-0.32,
        fontprops=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1,
    )

    # legend: method colours + segment shading, all in one block
    method_handles = [
        mpatches.Patch(facecolor=CONFIGDICT_COLOURS[k], label=FANCYDICT[k]) for k in x
    ]
    segment_handles = [
        mpatches.Patch(facecolor="#555555",   label="Other clusters"),
        mpatches.Patch(facecolor="#555555BB", label="Pairs"),
        mpatches.Patch(facecolor="#55555566", label="Singletons"),
    ]
    plt.legend(
        handles=method_handles,
        labels=[h.get_label() for h in method_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.46),   # below the x-axis labels and the bracket
        frameon=False,
        prop=ibmplexsans,
        handlelength=0.8,
        handletextpad=0.75,
        labelspacing=0.3,
        ncol=len(method_handles),       # all in one row
    )
    family_footnote = get_family_footnote(x)
    if family_footnote is not None:
        plt.text(
            0.5, -0.58, family_footnote,
            fontproperties=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1,
            horizontalalignment="center", verticalalignment="top", transform=ax.transAxes,
        )

    save_figure(fig, outfolder, "_".join(["plot_stackedbar", datatype, assembly, "n_clusters"]), bbox_inches="tight")
    fig.clf()
    del fig, ax


def number_of_clusters_stacked_bar_vs_c(theargs):
    print(f"[TRACE] >>> Entering number_of_clusters_stacked_bar_vs_c() - defined at line 3354 of {__file__}")
    """Same singleton/pair/3+ cluster-size breakdown as
    number_of_clusters_stacked_bar, but shown across the full sweep of
    sequence-identity thresholds `c` (one small multiple/subplot per `c`
    value, or bars grouped by `c` along the X axis -- see the plotting
    code below for the exact layout) rather than at just the default `c`.

    Figure interpretation:
        - Same segment meaning as number_of_clusters_stacked_bar
          (singleton / pair / 3+-member clusters), but repeated for every
          `c` threshold tested, so the reader can see how a method's
          cluster-size profile shifts as the identity cutoff is tightened
          or relaxed.
    Biological reading: as `c` increases (stricter identity requirement),
    a healthy method should show its singleton fraction grow gradually
    (genes on the edge of a family's identity range start falling out),
    not suddenly fragment a large fraction of previously well-formed
    clusters -- an abrupt jump in singleton/pair share at a particular
    `c` is a sign the method is unstable around that threshold for this
    dataset.

    Input/Output: same shape/behaviour as plotter (full `c` sweep, not
        fixed to the default).
    """
    name, datadf, namedict, outfolder, assembly, datatype, font_props = theargs
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props

    print(f"\t- Plotting stacked bar vs c for simulations of {namedict[assembly]}")
    subdf = datadf[
        (datadf.index.get_level_values("assembly") == assembly)
        & (datadf.index.get_level_values("simulations") == (datatype == "simulations"))
    ]

    if subdf.empty:
        warnings.warn(
            f"No rows available for {assembly}/{datatype}; skipping stacked bar vs c",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    xs = sorted(set(subdf["c"].astype(float)))
    x = []
    for combo in COMBO_ORDER:
        clusterer, seqtype = combo.split("/")
        if clusterer in SKETCH_METHOD_NAMES or clusterer in EMBED_METHOD_NAMES:
            continue
        if combo not in FANCYDICT or combo not in CONFIGDICT_COLOURS:
            continue
        if len(list(subdf[
            (subdf.index.get_level_values("clusterer") == clusterer)
            & (subdf["st"] == seqtype)
        ]["n_clusters"])):
            x.append(combo)

    if not x:
        warnings.warn(
            f"No clusterer/sequence-type rows available for {assembly}/{datatype}; skipping stacked bar vs c",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    n_methods = len(x)
    n_c = len(xs)
    bar_width = 0.8 / n_methods
    group_spacing = 1.0

    mean_total      = np.full((n_methods, n_c), np.nan)
    mean_singletons = np.full((n_methods, n_c), np.nan)
    mean_pairs      = np.full((n_methods, n_c), np.nan)
    mean_rest       = np.full((n_methods, n_c), np.nan)

    for m_idx, x_value in enumerate(x):
        for c_idx, c_value in enumerate(xs):
            tmpdf = subdf[
                (subdf.index.get_level_values("simulations") == (datatype == "simulations"))
                & (subdf.index.get_level_values("assembly") == assembly)
                & (subdf.index.get_level_values("clusterer") == x_value.split("/")[0])
                & (subdf["st"] == x_value.split("/")[1])
                & (subdf["c"] == c_value)
            ]
            if tmpdf.empty:
                continue
            t = tmpdf["n_clusters"].astype(float).mean()
            s = tmpdf["n_singletons"].astype(float).mean()
            p = tmpdf["n_pairs"].astype(float).mean()
            mean_total[m_idx, c_idx]      = t
            mean_singletons[m_idx, c_idx] = s
            mean_pairs[m_idx, c_idx]      = p
            mean_rest[m_idx, c_idx]       = t - s - p

    fig = plt.figure(1, dpi=150, figsize=(max(DEFAULT_FIGSIZE[0], n_c * n_methods * 0.65), DEFAULT_FIGSIZE[1]))
    ax = fig.subplots()

    # requirement 6: exact singleton/pair/rest/total counts vs c behind this plot
    write_metric_csv(
        pd.DataFrame({
            "method": [FANCYDICT[x[m_idx]] for m_idx in range(n_methods) for c_idx in range(n_c)],
            "combo": [x[m_idx] for m_idx in range(n_methods) for c_idx in range(n_c)],
            "c": [xs[c_idx] for m_idx in range(n_methods) for c_idx in range(n_c)],
            "mean_total_clusters": mean_total.flatten(),
            "mean_singleton_clusters": mean_singletons.flatten(),
            "mean_pair_clusters": mean_pairs.flatten(),
            "mean_other_clusters": mean_rest.flatten(),
        }),
        outfolder, "_".join(["plot_stackedbar_c", datatype, assembly, "n_clusters"]),
    )

    group_positions = np.arange(n_c) * group_spacing
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * bar_width

    import matplotlib.patches as mpatches
    for m_idx, x_value in enumerate(x):
        col            = CONFIGDICT_COLOURS[x_value]   # darkest  → other clusters (bottom)
        col_pairs      = col + "BB"                    # mid      → pairs
        col_singletons = col + "66"                    # lightest → singletons (top)
        bar_positions  = group_positions + offsets[m_idx]
        valid = ~np.isnan(mean_total[m_idx])

        # bottom: other clusters (darkest)
        ax.bar(bar_positions[valid], mean_rest[m_idx][valid], bar_width * 0.9, color=col)
        # middle: pairs
        ax.bar(bar_positions[valid], mean_pairs[m_idx][valid], bar_width * 0.9,
               bottom=mean_rest[m_idx][valid], color=col_pairs)
        # top: singletons (lightest)
        ax.bar(bar_positions[valid], mean_singletons[m_idx][valid], bar_width * 0.9,
               bottom=(mean_rest[m_idx] + mean_pairs[m_idx])[valid], color=col_singletons)

        for pos, total, v in zip(bar_positions, mean_total[m_idx], valid):
            if v:
                ax.text(
                    pos, total * 1.03,
                    f"{int(round(total))}",
                    ha="center", va="bottom",
                    fontproperties=ibmplexsans,
                    fontsize=7,
                    rotation=90,
                )

    ax.set_xlim(-0.6, (n_c - 1) * group_spacing + 0.6)
    ax.set_ylim(0, np.nanmax(mean_total) * 1.2)
    ax.set_xticks(group_positions)
    ax.set_xticklabels([str(c) for c in xs])
    ax.set_xlabel("minimum sequence identity (adim.)", fontproperties=ibmplexsans, loc="right", fontsize=AXIS_TITLE_FONT_SIZE)
    ax.set_ylabel("Number of clusters (adim.)", fontproperties=ibmplexsans, loc="top", fontsize=AXIS_TITLE_FONT_SIZE)

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="major", direction="in")
    ax.tick_params(which="minor", direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.get_yaxis().get_offset_text().set_x(-0.075)
    ax.get_yaxis().get_offset_text().set_fontproperties(ibmplexsans)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)

    plt.text(0, 1.01, namedict[assembly], fontproperties=ibmplexsansitalics,
             horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)

    method_handles = [
        mpatches.Patch(facecolor=CONFIGDICT_COLOURS[k], label=FANCYDICT[k]) for k in x
    ]
    segment_handles = [
        mpatches.Patch(facecolor="#555555",   label="Other clusters"),
        mpatches.Patch(facecolor="#555555BB", label="Pairs"),
        mpatches.Patch(facecolor="#55555566", label="Singletons"),
    ]
    plt.legend(
        handles=method_handles,
        labels=[h.get_label() for h in method_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),   # below the x-axis
        frameon=False,
        prop=ibmplexsans,
        handlelength=0.8,
        handletextpad=0.75,
        labelspacing=0.3,
        ncol=len(method_handles),       # all in one row; reduce to 3 if too wide
    )
    save_figure(fig, outfolder, "_".join(["plot_stackedbar_c", datatype, assembly, "n_clusters"]), bbox_inches="tight")
    fig.clf()
    del fig, ax


def methods_comparison_heatmap(theargs):
    print(f"[TRACE] >>> Entering methods_comparison_heatmap() - defined at line 3526 of {__file__}")
    """Heatmap comparing all clustering methods side by side across the main
    agreement-with-truth metrics (mean over seeds, at the default c). This is
    the 'contingency-style' comparison view: rows = methods (in the requested
    order), columns = metrics, colour = mean score.

    Figure interpretation:
        - Rows: one per (clusterer, sequence-type) combination, in
          COMBO_ORDER.
        - Columns: the four core truth-agreement metrics -- Adjusted Rand
          index, purity, adjusted mutual information, V-measure -- each
          in [0, 1] with 1 meaning perfect agreement with the simulated
          ground truth.
        - Cell colour/annotated value: the mean of that metric over all
          seed replicates for that method at the default sequence-
          identity threshold; brighter/darker shading follows the
          colourmap used below to make strong vs. weak performers
          visually obvious across all four metrics at once.
    Biological reading: a row that is uniformly bright across all four
    columns is a method that reliably recovers the true simulated gene
    families from every angle (pairwise agreement, purity, information
    content, and homogeneity/completeness balance); a row that is bright
    on some columns but dark on others indicates a method with a
    specific failure mode (e.g. good purity but poor completeness would
    mean it is over-splitting true families into many small, "pure"
    fragments rather than correctly merging them).

    Statistical significance: the Adjusted Rand index and Adjusted mutual
    information cells are annotated with "†" when that method's mean
    permutation-test p-value (see permutation_test_agreement, computed
    per-seed in calculate_values_from_cluster_matrix and averaged here
    across the same seed replicates as the cell's own mean score) is
    below SIGNIFICANCE_ALPHA -- i.e. that method's agreement with the
    simulated ground truth is unlikely to have arisen from chance label
    assignment. Purity and V-measure are never annotated (no p-value is
    computed for them; see calculate_values_from_cluster_matrix).
    """
    name, datadf, namedict, outfolder, assembly, datatype, font_props = theargs
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props

    print(f"\t- Plotting method-comparison heatmap for simulations of {namedict[assembly]}")
    subdf = datadf[
        (datadf.index.get_level_values("assembly") == assembly)
        & (datadf.index.get_level_values("simulations") == (datatype == "simulations"))
        & (datadf["c"] == DEFAULT_PARAMS["c"])
    ]

    if subdf.empty:
        warnings.warn(
            f"No rows available for {assembly}/{datatype}; skipping method-comparison heatmap",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    metric_cols = ["adj_rand_index", "purity", "adj_mutual_info", "v_measure"]
    metric_labels = ["Adjusted Rand\nindex", "Purity", "Adjusted mutual\ninformation", "V-measure"]

    x = []
    for combo in COMBO_ORDER:
        clusterer, seqtype = combo.split("/")
        if combo not in FANCYDICT:
            continue
        if len(subdf[
            (subdf.index.get_level_values("clusterer") == clusterer)
            & (subdf["st"] == seqtype)
        ]):
            x.append(combo)

    if not x:
        warnings.warn(
            f"No clusterer/sequence-type rows available for {assembly}/{datatype}; "
            "skipping method-comparison heatmap",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    # p-value column paired with each metric_col that has one (ARI, AMI);
    # metric_cols without an entry here (purity, v_measure) are never
    # significance-annotated below.
    metric_pvalue_cols = {
        "adj_rand_index": ADJ_RAND_INDEX_PVALUE_COL,
        "adj_mutual_info": ADJ_MUTUAL_INFO_PVALUE_COL,
    }

    mat = np.full((len(x), len(metric_cols)), np.nan)
    pmat = np.full((len(x), len(metric_cols)), np.nan)  # mean p-value, where applicable
    for i, combo in enumerate(x):
        clusterer, seqtype = combo.split("/")
        tmpdf = subdf[
            (subdf.index.get_level_values("clusterer") == clusterer)
            & (subdf["st"] == seqtype)
        ]
        for j, metric_col in enumerate(metric_cols):
            mat[i, j] = tmpdf[metric_col].astype(float).mean()
            pvalue_col = metric_pvalue_cols.get(metric_col)
            if pvalue_col is not None:
                pmat[i, j] = tmpdf[pvalue_col].astype(float).mean()

    row_labels = [
        FANCYDICT[c] + (" *" if c.split("/")[0] in SKETCH_METHOD_NAMES or c.split("/")[0] in EMBED_METHOD_NAMES else "") for c in x
    ]

    # export the exact mean values (and, for ARI/AMI, mean p-values) behind
    # every cell of this heatmap
    write_metric_csv(
        pd.DataFrame({
            "method": [FANCYDICT[c] for c in x for _ in metric_cols],
            "combo": [c for c in x for _ in metric_cols],
            "metric": [metric_col for _ in x for metric_col in metric_cols],
            "mean": mat.flatten(),
            "mean_pvalue": pmat.flatten(),
        }),
        outfolder, "_".join(["plot_heatmap_methodcomparison", datatype, assembly]),
    )

    fig = plt.figure(
        1, dpi=150,
        figsize=(max(6.0, len(metric_cols) * 1.6 + 2.0), max(4.0, len(x) * 0.42 + 1.5)),
    )
    ax = fig.subplots()
    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)

    im = ax.imshow(
        mat,
        cmap="YlGnBu",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(x)))
    ax.set_yticklabels(row_labels)

    any_pvalue_annotated = False
    for i in range(len(x)):
        for j in range(len(metric_cols)):
            val = mat[i, j]
            if np.isnan(val):
                continue
            txt_color = "white" if val < 0.5 else "black"
            cell_text = f"{val:.2f}"
            pval = pmat[i, j]
            if not np.isnan(pval) and pval < SIGNIFICANCE_ALPHA:
                cell_text += "†"
                any_pvalue_annotated = True
            ax.text(
                j, i, cell_text,
                ha="center", va="center",
                fontsize=BASE_FONT_SIZE, color=txt_color,
                fontproperties=ibmplexsans,
            )

    ax.set_xticks(np.arange(len(metric_cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(x) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", bottom=False, left=False)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Mean score vs ground truth (adim.)", fontproperties=ibmplexsans)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)

    plt.text(0, 1.03, namedict[assembly], fontproperties=ibmplexsansitalics,
             horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)

    footnote_lines = []
    family_footnote = get_family_footnote(x)
    if family_footnote is not None:
        footnote_lines.append(family_footnote)
    if any_pvalue_annotated:
        footnote_lines.append(
            f"† mean permutation-test p < {SIGNIFICANCE_ALPHA:g} vs ground truth "
            "(ARI/AMI only; see permutation_test_agreement)"
        )
    if footnote_lines:
        plt.text(
            0.5, -0.14, "\n".join(footnote_lines),
            fontproperties=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1,
            horizontalalignment="center", verticalalignment="top", transform=ax.transAxes,
        )

    save_figure(fig, outfolder, "_".join(["plot_heatmap_methodcomparison", datatype, assembly]), bbox_inches="tight")
    fig.clf()
    del fig, ax


def build_pairwise_ari_matrix(combo_list, labels_by_seed):
    print(f"[TRACE] >>> Entering build_pairwise_ari_matrix() - defined at line 3672 of {__file__}")
    """Compute the pairwise Adjusted Rand Index (ARI) between every pair
    of clustering methods (not against ground truth -- method vs.
    method), averaged over random-seed replicates, for the pairwise ARI
    heatmap.

    Input:
        combo_list    -- ordered list of "clusterer/seqtype" combo keys
                          (e.g. "cdhit/aa") to include, in display order.
        labels_by_seed -- {seed: {combo: {gene_id: cluster_label}}},
                          i.e. each method's per-gene cluster assignment,
                          per seed.
    Output: an (n x n) numpy array `mat` where n = len(combo_list) and
        mat[j, i] (j > i) holds the seed-averaged ARI between combo_list[i]
        and combo_list[j], computed only over genes both methods actually
        assigned to a cluster (their common gene set). ARI ranges from
        ~0 (no better than random agreement) to 1 (identical clusterings).

    Why only the lower triangle: ARI(i, j) == ARI(j, i) (the metric is
    symmetric), so we only ever compute and store the lower triangle
    (mat[j, i] with j > i). The upper triangle is left as NaN on purpose;
    plot_pairwise_ari_heatmap masks NaNs to white so only a lower-triangle
    heatmap is drawn instead of the redundant full grid.
    """
    n = len(combo_list)
    mat = np.full((n, n), np.nan)

    for i, combo_i in enumerate(combo_list):
        mat[i, i] = 1.0
        for j in range(i + 1, len(combo_list)):
            combo_j = combo_list[j]
            seed_aris = []
            for seed, combo_dict in labels_by_seed.items():
                if combo_i not in combo_dict or combo_j not in combo_dict:
                    continue
                labels_i = combo_dict[combo_i]
                labels_j = combo_dict[combo_j]

                # Match on genes both methods actually assigned to a cluster.
                common_genes = [g for g in labels_i if g in labels_j]
                if len(common_genes) < 2:
                    continue

                ari = metrics.adjusted_rand_score(
                    [labels_i[g] for g in common_genes],
                    [labels_j[g] for g in common_genes],
                )
                seed_aris.append(float(ari))

            if seed_aris:
                mat[j, i] = float(np.mean(seed_aris))

    return mat


def _pairwise_purity_score(labels_i, labels_j):
    print(f"[TRACE] >>> Entering _pairwise_purity_score() - defined at line 3727 of {__file__}")
    """Symmetric purity between two clusterings. Purity itself is not a
    symmetric measure (it depends on which clustering is treated as the
    'truth'), so for a method-vs-method comparison we average the score
    computed in both directions."""
    from sklearn.metrics.cluster import contingency_matrix

    cm = contingency_matrix(labels_i, labels_j)
    total = np.sum(cm)
    purity_i_as_truth = float(np.sum(np.amax(cm, axis=1))) / float(total)
    purity_j_as_truth = float(np.sum(np.amax(cm, axis=0))) / float(total)
    return (purity_i_as_truth + purity_j_as_truth) / 2.0


# Metric functions used by build_pairwise_metric_matrix below. Each takes
# two same-length label lists (already restricted to the genes common to
# both methods) and returns a single float agreement score.
PAIRWISE_METRIC_FUNCTIONS = {
    "ari": metrics.adjusted_rand_score,
    "ami": adjusted_mutual_info_score,
    "purity": _pairwise_purity_score,
    "v_measure": metrics.v_measure_score,
}


def build_pairwise_metric_matrix(combo_list, labels_by_seed, metric_name):
    print(f"[TRACE] >>> Entering build_pairwise_metric_matrix() - defined at line 3752 of {__file__}")
    """Generic version of build_pairwise_ari_matrix, parameterised by which
    agreement metric to use (see PAIRWISE_METRIC_FUNCTIONS). Matrix is
    symmetric, so only the lower triangle (mat[j, i] with j > i) is
    computed and stored; the upper triangle is left as NaN on purpose, and
    _plot_triangular_pairwise_heatmap masks NaNs to white so only a
    lower-triangle heatmap is drawn instead of the redundant full grid.
    """
    metric_function = PAIRWISE_METRIC_FUNCTIONS[metric_name]
    n = len(combo_list)
    mat = np.full((n, n), np.nan)

    for i, combo_i in enumerate(combo_list):
        mat[i, i] = 1.0
        for j in range(i + 1, len(combo_list)):
            combo_j = combo_list[j]
            seed_scores = []
            for seed, combo_dict in labels_by_seed.items():
                if combo_i not in combo_dict or combo_j not in combo_dict:
                    continue
                labels_i = combo_dict[combo_i]
                labels_j = combo_dict[combo_j]

                # Match on genes both methods actually assigned to a cluster.
                common_genes = [g for g in labels_i if g in labels_j]
                if len(common_genes) < 2:
                    continue

                score = metric_function(
                    [labels_i[g] for g in common_genes],
                    [labels_j[g] for g in common_genes],
                )
                seed_scores.append(float(score))

            if seed_scores:
                mat[j, i] = float(np.mean(seed_scores))

    return mat


def build_pairwise_f1_matrix(combo_list, labels_by_seed, total_genes_by_seed=None):
    print(f"[TRACE] >>> Entering build_pairwise_f1_matrix() - defined at line 3792 of {__file__}")
    """Pairwise agreement between methods on *which genes they kept*, now
    penalised for genes deleted with respect to the ORIGINAL gene set
    (requirement 1).

    === CHANGE (requirement 1: gene-deletion penalty) ===============
    Previously this only used the genes each method actually retained:

        F1_old = 2 * |kept_i ∩ kept_j| / (|kept_i| + |kept_j|)

    That is a fair "do these two methods agree with each other" score, but
    it does NOT penalise a method (or a pair of methods) for having deleted
    genes that existed in the original dataset: e.g. two methods that each
    keep a totally different, small, non-overlapping subset of the genome
    would already score 0 (correctly), but two methods that keep the SAME
    small subset would score a misleading 1.0, even though most of the
    original genome was lost by both of them.

    To fix this we bring in `total_genes_by_seed` — the number of genes N in
    the ORIGINAL dataset for that seed (see n_original_genes returned by
    get_info_from_folder) — and treat the comparison as a precision/recall
    problem against that original universe of N genes:

      - "positive" event = a gene from the original dataset that BOTH
        methods agreed to keep (kept_i ∩ kept_j). This is the only kind of
        "positive" available since a method can only keep genes that were
        present in the original dataset in the first place (no method can
        invent new genes), so there are no possible false positives.
      - TP = |kept_i ∩ kept_j|
      - FP = 0 (a kept gene is by construction a subset of the original N)
      - FN = N - TP  (every original gene that was NOT kept by both methods,
        i.e. that was deleted by at least one of the two methods)
      - Precision = TP / (TP + FP) = 1
      - Recall    = TP / (TP + FN) = TP / N
      - F1 = 2 * Precision * Recall / (Precision + Recall)
           = 2 * TP / (N + TP)

    Assumption: N (the size of the original gene set) is taken to be the
    same reference for both methods being compared in a given seed — i.e.
    the ground-truth gene count for that assembly/seed, not the union or
    intersection of what either method happened to output. This is what
    makes the score a genuine "penalty for deletion" rather than just an
    agreement score: the more genes either method deletes, the smaller TP
    gets relative to the fixed N, and the lower the score, even if the two
    methods agree perfectly on the (small) subset they did keep.

    If `total_genes_by_seed` is not supplied, this falls back to the
    original, non-penalised Dice/F1 formula (kept for backward
    compatibility / testing).

    This remains symmetric in i/j (no method is "ground truth"), so exactly
    like the ARI matrix we only need to compute and store the lower
    triangle.
    """
    n = len(combo_list)
    mat = np.full((n, n), np.nan)

    for i, combo_i in enumerate(combo_list):
        mat[i, i] = 1.0
        for j in range(i + 1, len(combo_list)):
            combo_j = combo_list[j]
            seed_f1s = []
            for seed, combo_dict in labels_by_seed.items():
                if combo_i not in combo_dict or combo_j not in combo_dict:
                    continue
                kept_i = set(combo_dict[combo_i].keys())
                kept_j = set(combo_dict[combo_j].keys())
                tp = len(kept_i & kept_j)

                # === CHANGE: penalised formula using the original gene
                # count N for this seed, when available ===
                n_original = None
                if total_genes_by_seed is not None:
                    n_original = total_genes_by_seed.get(seed)

                if n_original:
                    denom = n_original + tp
                    if denom == 0:
                        continue
                    f1 = 2 * tp / denom
                else:
                    # fallback: old, non-penalised behaviour
                    denom = len(kept_i) + len(kept_j)
                    if denom == 0:
                        continue
                    f1 = 2 * tp / denom

                seed_f1s.append(float(f1))

            if seed_f1s:
                mat[j, i] = float(np.mean(seed_f1s))

    return mat


def build_pairwise_f1_matrix_with_additions(
    combo_list, labels_by_seed, total_genes_by_seed=None,
    n_added_by_seed=None, mode="deleted_only",
):
    print(f"[TRACE] >>> Entering build_pairwise_f1_matrix_with_additions() - defined at line 3887 of {__file__}")
    """See the module-level comment directly above for the three `mode`
    options and their formulas. `n_added_by_seed` should have the same
    shape as `total_genes_by_seed` but per-combo: {seed: {combo: n_added}}
    (this is exactly `addition_by_assembly[assembly]` as built in main()).
    """
    if mode not in ("deleted_only", "added_as_fp", "net_kept"):
        raise ValueError(f"Unknown mode {mode!r}; expected one of "
                          "'deleted_only', 'added_as_fp', 'net_kept'")

    n = len(combo_list)
    mat = np.full((n, n), np.nan)

    for i, combo_i in enumerate(combo_list):
        mat[i, i] = 1.0
        for j in range(i + 1, len(combo_list)):
            combo_j = combo_list[j]
            seed_f1s = []
            for seed, combo_dict in labels_by_seed.items():
                if combo_i not in combo_dict or combo_j not in combo_dict:
                    continue
                kept_i = set(combo_dict[combo_i].keys())
                kept_j = set(combo_dict[combo_j].keys())
                tp = len(kept_i & kept_j)

                n_original = total_genes_by_seed.get(seed) if total_genes_by_seed else None
                added_i = (n_added_by_seed or {}).get(seed, {}).get(combo_i, 0)
                added_j = (n_added_by_seed or {}).get(seed, {}).get(combo_j, 0)

                if not n_original:
                    denom = len(kept_i) + len(kept_j)
                    if denom == 0:
                        continue
                    seed_f1s.append(float(2 * tp / denom))
                    continue

                if mode == "deleted_only":
                    denom = n_original + tp
                    if denom == 0:
                        continue
                    f1 = 2 * tp / denom
                elif mode == "added_as_fp":
                    fp = added_i + added_j
                    fn = n_original - tp
                    denom_p = tp + fp
                    denom_r = tp + fn
                    precision = (tp / denom_p) if denom_p else 0.0
                    recall = (tp / denom_r) if denom_r else 0.0
                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if (precision + recall) else 0.0
                    )
                else:  # mode == "net_kept"
                    tp_eff = max(0, tp - added_i - added_j)
                    denom = n_original + tp_eff
                    f1 = (2 * tp_eff / denom) if denom else 0.0

                seed_f1s.append(float(f1))

            if seed_f1s:
                mat[j, i] = float(np.mean(seed_f1s))

    return mat


def build_pairwise_exact_match_matrix(combo_list, labels_by_seed):
    print(f"[TRACE] >>> Entering build_pairwise_exact_match_matrix() - defined at line 3955 of {__file__}")
    """Row-normalised agreement matrix: mat[i, j] is the fraction of tool
    i's clusters that have an EXACT match (identical gene membership set)
    among tool j's clusters, averaged over seeds (real data only ever has
    one pseudo-seed, "run").

    Unlike the ARI/AMI/purity/F1 pairwise matrices above, this is NOT
    symmetric in general (tool i and tool j can have different numbers of
    clusters, so "fraction of i's clusters matched in j" need not equal
    "fraction of j's clusters matched in i"), so the full n x n matrix is
    computed and returned (no upper-triangle NaN masking).

    A cluster is represented as the frozenset of gene ids it contains
    (restricted, like every other pairwise metric here, to genes both
    tools actually assigned to a cluster is NOT needed here -- an exact
    match is compared against the tool's own full gene set for that
    cluster, since partial-membership genes would trivially break an
    "exact" match anyway).
    """
    n = len(combo_list)
    mat = np.full((n, n), np.nan)

    for i, combo_i in enumerate(combo_list):
        mat[i, i] = 1.0
        for j, combo_j in enumerate(combo_list):
            if i == j:
                continue
            seed_fracs = []
            for seed, combo_dict in labels_by_seed.items():
                if combo_i not in combo_dict or combo_j not in combo_dict:
                    continue
                labels_i = combo_dict[combo_i]
                labels_j = combo_dict[combo_j]

                clusters_i = {}
                for gene, label in labels_i.items():
                    clusters_i.setdefault(label, set()).add(gene)
                clusters_j_sets = set()
                tmp_j = {}
                for gene, label in labels_j.items():
                    tmp_j.setdefault(label, set()).add(gene)
                for members in tmp_j.values():
                    clusters_j_sets.add(frozenset(members))

                if not clusters_i:
                    continue

                n_matched = sum(
                    1 for members in clusters_i.values()
                    if frozenset(members) in clusters_j_sets
                )
                seed_fracs.append(n_matched / len(clusters_i))

            if seed_fracs:
                mat[i, j] = float(np.mean(seed_fracs))

    return mat


def plot_pairwise_exact_match_heatmap(theargs):
    print(f"[TRACE] >>> Entering plot_pairwise_exact_match_heatmap() - defined at line 4014 of {__file__}")
    """Full (non-triangular) heatmap of pairwise exact-cluster-match
    agreement between pan-genome clustering tools: row = "query" tool,
    column = "reference" tool, cell = fraction of the row tool's clusters
    that are an exact gene-membership match to one of the column tool's
    clusters (see build_pairwise_exact_match_matrix)."""
    labels_by_seed, namedict, outfolder, assembly, datatype, font_props = theargs
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props

    print(f"\t- Plotting pairwise exact-cluster-match heatmap for {namedict[assembly]}")

    if not labels_by_seed:
        warnings.warn(
            f"No per-method label data available for {assembly}/{datatype}; "
            "skipping pairwise exact-match heatmap",
            RuntimeWarning, stacklevel=2,
        )
        return

    combos_present = set()
    for combo_dict in labels_by_seed.values():
        combos_present.update(combo_dict.keys())

    x = [combo for combo in COMBO_ORDER if combo in combos_present and combo in FANCYDICT]

    if not x:
        warnings.warn(
            f"No clusterer/sequence-type combos available for {assembly}/{datatype}; "
            "skipping pairwise exact-match heatmap",
            RuntimeWarning, stacklevel=2,
        )
        return

    mat = build_pairwise_exact_match_matrix(x, labels_by_seed)
    labels = [FANCYDICT[c] for c in x]

    fig = plt.figure(
        1, dpi=150,
        figsize=(max(6.0, len(x) * 0.5 + 2.0), max(6.0, len(x) * 0.5 + 2.0)),
    )
    ax = fig.subplots()

    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(x)))
    ax.set_yticklabels(labels)

    for i in range(len(x)):
        for j in range(len(x)):
            val = mat[i, j]
            if np.isnan(val):
                continue
            txt_color = "white" if val > 0.5 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=BASE_FONT_SIZE, color=txt_color,
                fontproperties=ibmplexsans,
            )

    ax.set_xticks(np.arange(len(x) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(x) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", bottom=False, left=False)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)

    ax.set_xlabel("Reference tool (matched against)", fontsize=AXIS_TITLE_FONT_SIZE, fontproperties=ibmplexsans)
    ax.set_ylabel("Query tool (fraction of its clusters matched)", fontsize=AXIS_TITLE_FONT_SIZE, fontproperties=ibmplexsans)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Fraction of exactly-matching clusters (adim.)", fontproperties=ibmplexsans)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)

    plt.text(0, 1.03, namedict[assembly], fontproperties=ibmplexsansitalics,
             horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)

    # Note: this heatmap is NOT symmetric (row = query tool, column =
    # reference tool matched against, and "query matches reference" need
    # not equal "reference matches query"), so per item 3 of the pipeline
    # unification (trees only for symmetric/similarity matrices) no NJ
    # tree is generated here.
    save_figure(fig, outfolder, "_".join(["plot_heatmap_pairwise_exact_match", datatype, assembly]), bbox_inches="tight")

    fig.clf()
    del fig, ax


def extract_strain_from_geneid(gene_id):
    print(f"[TRACE] >>> Entering extract_strain_from_geneid() - defined at line 4107 of {__file__}")
    """Heuristic extraction of the strain/isolate identifier a gene id
    belongs to. Real-data gene ids here are locus tags of the form
    "<STRAIN_PREFIX>_<NUMBER>" (the standard Prokka/Panaroo locus-tag
    convention, e.g. "NLEIDEKG_01145"), optionally with a
    "_refound_<N>" suffix appended by Panaroo. There is no per-gene
    strain field carried through the cluster matrices built earlier in
    this pipeline, so the strain is recovered directly from the gene id
    string:
      1. Strip a trailing "_refound_<N>" suffix, if present.
      2. Take everything before the final "_<digits>" suffix as the
         strain identifier.
      3. If the id doesn't match that pattern at all, fall back to using
         the whole id as its own "strain" (keeps the function total and
         gives a graceful, conservative degradation rather than an error).
    """
    gid = str(gene_id)
    gid = re.sub(r"_refound_\d+$", "", gid)
    match = re.match(r"^(.*)_\d+$", gid)
    if match:
        return match.group(1)
    return gid


def compute_cluster_strain_counts(gene_label_dict):
    print(f"[TRACE] >>> Entering compute_cluster_strain_counts() - defined at line 4131 of {__file__}")
    """Given one tool's {gene_id: cluster_label} dict, return a list with
    the number of distinct strains represented in each cluster (one entry
    per cluster, unsorted)."""
    clusters = {}
    for gene, label in gene_label_dict.items():
        clusters.setdefault(label, set()).add(extract_strain_from_geneid(gene))
    return [len(strains) for strains in clusters.values()]


def estimate_total_strains(labels_by_seed):
    print(f"[TRACE] >>> Entering estimate_total_strains() - defined at line 4141 of {__file__}")
    """Approximate the total number of strains/isolates in the real
    dataset as the union of strains inferred (via
    extract_strain_from_geneid) from every gene seen across every tool
    and seed. Since every tool clusters genes drawn from the same set of
    input genomes, this union should closely approximate the true isolate
    count even though individual tools may not have a representative from
    every strain in every cluster."""
    strains = set()
    for combo_dict in labels_by_seed.values():
        for gene_label_dict in combo_dict.values():
            for gene in gene_label_dict:
                strains.add(extract_strain_from_geneid(gene))
    return len(strains)


def plot_core_genome_curve_realdata(theargs):
    print(f"[TRACE] >>> Entering plot_core_genome_curve_realdata() - defined at line 4157 of {__file__}")
    """Core-genome estimation plot (real data only): for each tool, sort
    its clusters in descending order by the number of distinct strains
    they contain, and plot that count against the cluster's rank. The
    point where a tool's curve drops below the total number of strains in
    the dataset marks its estimated core-genome size (the number of
    clusters present in essentially every strain). Follows the same
    "core genome estimation" style as standard pan-genome-tool benchmark
    plots (step curve per tool, dashed horizontal line at the total
    strain count)."""
    labels_by_seed, namedict, outfolder, assembly, datatype, font_props = theargs
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props

    print(f"\t- Plotting core genome estimation curve for {namedict[assembly]}")

    if not labels_by_seed:
        warnings.warn(
            f"No per-method label data available for {assembly}/{datatype}; "
            "skipping core genome estimation curve",
            RuntimeWarning, stacklevel=2,
        )
        return

    combos_present = set()
    for combo_dict in labels_by_seed.values():
        combos_present.update(combo_dict.keys())

    x = [combo for combo in COMBO_ORDER if combo in combos_present and combo in FANCYDICT]

    if not x:
        warnings.warn(
            f"No clusterer/sequence-type combos available for {assembly}/{datatype}; "
            "skipping core genome estimation curve",
            RuntimeWarning, stacklevel=2,
        )
        return

    total_strains = estimate_total_strains(labels_by_seed)
    if total_strains == 0:
        warnings.warn(
            f"Could not infer any strain identifiers for {assembly}/{datatype}; "
            "skipping core genome estimation curve",
            RuntimeWarning, stacklevel=2,
        )
        return

    # Merge across seeds (real data only ever has one, "run") by combo.
    curves = {}
    for combo in x:
        counts = []
        for combo_dict in labels_by_seed.values():
            if combo in combo_dict:
                counts.extend(compute_cluster_strain_counts(combo_dict[combo]))
        if counts:
            curves[combo] = sorted(counts, reverse=True)

    if not curves:
        warnings.warn(
            f"No cluster/strain data available for {assembly}/{datatype}; "
            "skipping core genome estimation curve",
            RuntimeWarning, stacklevel=2,
        )
        return

    fig = plt.figure(1, dpi=150, figsize=DEFAULT_FIGSIZE)
    ax = fig.subplots()

    for combo in x:
        if combo not in curves:
            continue
        counts = curves[combo]
        ranks = np.arange(1, len(counts) + 1)
        color = CONFIGDICT_COLOURS.get(combo)
        # Core genome size = number of leading clusters at (or above) the
        # full strain count, i.e. where the curve has not yet dropped
        # below the total number of strains.
        core_size = sum(1 for c in counts if c >= total_strains)
        label = f"{FANCYDICT[combo]}. core: {core_size} total: {len(counts)}"
        ax.step(ranks, counts, where="post", label=label, color=color, linewidth=1.3)

    # requirement 6: exact per-cluster strain counts + core-genome size
    # summary behind this curve
    write_metric_csv(
        pd.DataFrame({
            "method": [FANCYDICT[combo] for combo in x if combo in curves for _ in curves[combo]],
            "combo": [combo for combo in x if combo in curves for _ in curves[combo]],
            "cluster_rank": [r for combo in x if combo in curves for r in range(1, len(curves[combo]) + 1)],
            "strains_in_cluster": [c for combo in x if combo in curves for c in curves[combo]],
            "total_strains": total_strains,
        }),
        outfolder, "_".join(["plot_core_genome_estimation", datatype, assembly]),
    )

    ax.axhline(total_strains, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("cluster rank", fontsize=AXIS_TITLE_FONT_SIZE, fontproperties=ibmplexsans)
    ax.set_ylabel("number of strains in cluster", fontsize=AXIS_TITLE_FONT_SIZE, fontproperties=ibmplexsans)
    ax.set_ylim(bottom=0, top=total_strains * 1.02)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right", fontsize=BASE_FONT_SIZE, frameon=True)
    ax.set_title(
        f"Core genome estimation — {namedict[assembly]}{get_datatype_title_suffix(datatype)}",
        fontproperties=ibmplexsansbold,
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)
    fig.tight_layout()

    save_figure(fig, outfolder, "_".join(["plot_core_genome_estimation", datatype, assembly]))
    plt.close(fig)


def compute_cluster_occupancy_percentages(gene_label_dict, total_strains):
    print(f"[TRACE] >>> Entering compute_cluster_occupancy_percentages() - defined at line 4255 of {__file__}")
    """Given one tool's {gene_id: cluster_label} dict and the total number
    of strains/isolates in the dataset (see estimate_total_strains), return
    a list with one entry per cluster: the percentage of all strains that
    are represented in that cluster (100 * distinct strains in cluster /
    total_strains). Reuses extract_strain_from_geneid, exactly like
    compute_cluster_strain_counts above, so it stays consistent with the
    core genome estimation curve's notion of "which strain a gene belongs
    to"."""
    if total_strains == 0:
        return []
    clusters = {}
    for gene, label in gene_label_dict.items():
        clusters.setdefault(label, set()).add(extract_strain_from_geneid(gene))
    return [100.0 * len(strains) / total_strains for strains in clusters.values()]


def compute_adaptive_occupancy_bins(pooled_percentages, n_bins=50):
    print(f"[TRACE] >>> Entering compute_adaptive_occupancy_bins() - defined at line 4272 of {__file__}")
    """Classic equal-width bin edges (0-100 %) for the cluster-occupancy
    histogram (the standard "frequency distribution of gene clusters
    across isolates" plot used e.g. for pangenome core/accessory
    breakdowns).

    Unlike a quantile/adaptive scheme, every bin is exactly the same
    width (100 / n_bins percentage points, 2 % wide by default), so the
    shape of the distribution -- including how sharply clusters pile up
    right at 0% and 100% -- is shown faithfully instead of being smeared
    out by wide bins in the sparse middle. Bins are always anchored at 0
    and 100 so every plot using this function shares the same edges and
    stays directly comparable.
    """
    return np.linspace(0.0, 100.0, n_bins + 1)


def compute_occupancy_by_combo(labels_by_seed, x, total_strains):
    print(f"[TRACE] >>> Entering compute_occupancy_by_combo() - defined at line 4289 of {__file__}")
    """Shared helper for the U-plot functions: merges per-seed label dicts
    (real data only ever has one, "run") into {combo: [percentages]},
    exactly the same merge plot_core_genome_curve_realdata does for strain
    counts. Used by both plot_cluster_occupancy_uplot (all methods
    overlaid) and plot_cluster_occupancy_uplot_single (one figure per
    method) so the two stay consistent with each other."""
    occupancy = {}
    for combo in x:
        percentages = []
        for combo_dict in labels_by_seed.values():
            if combo in combo_dict:
                percentages.extend(
                    compute_cluster_occupancy_percentages(combo_dict[combo], total_strains)
                )
        if percentages:
            occupancy[combo] = percentages
    return occupancy



# Strain-prevalence ranges used both by the U-plot category-count CSV and
# the pairwise Jensen-Shannon divergence CSV (requirements 2 and 3).
# Bounds follow the standard pangenome core/shell/cloud convention: core
# (95-100% of strains), shell (15-95%), cloud (0-15%). Edges are
# inclusive of their upper bound and exclusive of their lower bound
# except for the first bin, which includes 0%, so every cluster falls
# into exactly one category.
OCCUPANCY_CATEGORY_EDGES = [
    ("clusters_0_15", 0.0, 15.0),
    ("clusters_15_95", 15.0, 95.0),
    ("clusters_95_100", 95.0, 100.0),
]


def compute_occupancy_category_counts(occupancy, x):
    print(f"[TRACE] >>> Entering compute_occupancy_category_counts() - defined near line 4500 of {__file__}")
    """For every method (combo) present in `occupancy` ({combo: [cluster
    occupancy percentages]}, as built by compute_occupancy_by_combo --
    the exact same data used to draw the U plots), count how many
    clusters fall into each of the three strain-prevalence ranges defined
    by OCCUPANCY_CATEGORY_EDGES (0-15%, 15-95%, 95-100% of strains).

    Input:  occupancy -- {combo: [percentages]} dict, as returned by
                          compute_occupancy_by_combo.
            x          -- ordered list of combo keys, controlling row
                          order in the returned DataFrame.
    Output: pandas DataFrame with one row per method and columns
            "method", "clusters_0_15", "clusters_15_95",
            "clusters_95_100" (counts, plus the FANCYDICT display label
            for readability).
    """
    rows = []
    for combo in x:
        if combo not in occupancy:
            continue
        percentages = np.asarray(occupancy[combo], dtype=float)
        row = {"method": FANCYDICT.get(combo, combo), "combo": combo}
        for col_name, lo, hi in OCCUPANCY_CATEGORY_EDGES:
            if lo == 0.0:
                mask = (percentages >= lo) & (percentages <= hi)
            else:
                mask = (percentages > lo) & (percentages <= hi)
            row[col_name] = int(mask.sum())
        rows.append(row)
    # column order matches the ranges low->high as requested in the prompt
    return pd.DataFrame(rows, columns=["method", "combo", "clusters_95_100", "clusters_15_95", "clusters_0_15"])


def compute_occupancy_jsd_matrix(occupancy, x):
    print(f"[TRACE] >>> Entering compute_occupancy_jsd_matrix() - defined near line 4500 of {__file__}")
    """Pairwise Jensen-Shannon divergence (base-2, bounded [0, 1]) between
    every pair of methods' distributions over the same three
    strain-prevalence categories used by compute_occupancy_category_counts
    (0-15%, 15-95%, 95-100% of strains) -- i.e. the same underlying data
    as the U plots, reduced to the 3-category distribution each method's
    clusters fall into.

    Input:  occupancy -- {combo: [percentages]} dict (see
                          compute_occupancy_by_combo).
            x          -- ordered list of combo keys, controlling row/
                          column order in the returned matrix.
    Output: pandas DataFrame, a symmetric (n x n) matrix of JSD values
            with method display labels as both the index and the columns
            (diagonal is 0.0, since JSD(P, P) = 0).
    """
    combos = [combo for combo in x if combo in occupancy]
    category_cols = [name for name, _, _ in OCCUPANCY_CATEGORY_EDGES]
    counts_df = compute_occupancy_category_counts(occupancy, combos).set_index("combo")

    labels = [FANCYDICT.get(combo, combo) for combo in combos]
    n = len(combos)
    jsd_mat = np.zeros((n, n), dtype=float)
    for i, combo_i in enumerate(combos):
        dist_i = counts_df.loc[combo_i, category_cols].to_numpy(dtype=float)
        for j, combo_j in enumerate(combos):
            if j < i:
                jsd_mat[i, j] = jsd_mat[j, i]
                continue
            dist_j = counts_df.loc[combo_j, category_cols].to_numpy(dtype=float)
            jsd_mat[i, j] = 0.0 if i == j else jensen_shannon_divergence(dist_i, dist_j, base=2.0)

    return pd.DataFrame(jsd_mat, index=labels, columns=labels)


def plot_cluster_occupancy_uplot(theargs):
    print(f"[TRACE] >>> Entering plot_cluster_occupancy_uplot() - defined at line 4309 of {__file__}")
    """Cluster-occupancy "U plot" (real data only): for each tool, the
    distribution of clusters across "percentage of isolates/strains
    represented in the cluster" (x-axis), with the y-axis giving the
    number of clusters falling into each percentage bin. A cluster made
    up of genes from every isolate contributes to the 100% end; a cluster
    drawn from a single isolate contributes near the 0% end. Bin edges are
    fixed-width, 2 percentage points each (see compute_adaptive_occupancy_bins)
    and shared across all tools so the curves stay directly comparable, and
    are computed on the default sequence-identity threshold `c` only
    (same DEFAULT_PARAMS["c"]-filtered combos as every other real-data
    plot in this pipeline, since labels_by_seed is already restricted to
    that when it's built upstream in get_info_from_folder_realdata)."""
    labels_by_seed, namedict, outfolder, assembly, datatype, font_props = theargs
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props

    print(f"\t- Plotting cluster-occupancy U plot for {namedict[assembly]}")

    if not labels_by_seed:
        warnings.warn(
            f"No per-method label data available for {assembly}/{datatype}; "
            "skipping cluster-occupancy U plot",
            RuntimeWarning, stacklevel=2,
        )
        return

    combos_present = set()
    for combo_dict in labels_by_seed.values():
        combos_present.update(combo_dict.keys())

    x = [combo for combo in COMBO_ORDER if combo in combos_present and combo in FANCYDICT]

    if not x:
        warnings.warn(
            f"No clusterer/sequence-type combos available for {assembly}/{datatype}; "
            "skipping cluster-occupancy U plot",
            RuntimeWarning, stacklevel=2,
        )
        return

    total_strains = estimate_total_strains(labels_by_seed)
    if total_strains == 0:
        warnings.warn(
            f"Could not infer any strain identifiers for {assembly}/{datatype}; "
            "skipping cluster-occupancy U plot",
            RuntimeWarning, stacklevel=2,
        )
        return

    # Merge across seeds (real data only ever has one, "run") by combo,
    # exactly as plot_core_genome_curve_realdata does.
    occupancy = compute_occupancy_by_combo(labels_by_seed, x, total_strains)

    if not occupancy:
        warnings.warn(
            f"No cluster/strain data available for {assembly}/{datatype}; "
            "skipping cluster-occupancy U plot",
            RuntimeWarning, stacklevel=2,
        )
        return

    # requirement 2: one CSV, all methods, with the cluster counts in
    # each of the three strain-prevalence ranges -- computed from the
    # exact same `occupancy` data used to draw the U plot below, so the
    # CSV and the figure can never disagree.
    occupancy_counts_df = compute_occupancy_category_counts(occupancy, x)
    write_metric_csv(
        occupancy_counts_df.drop(columns=["combo"]), outfolder,
        "_".join(["cluster_occupancy_category_counts", datatype, assembly]),
    )

    # requirement 3: pairwise Jensen-Shannon divergence between methods'
    # distributions over those same three categories, as a method x
    # method matrix CSV.
    jsd_df = compute_occupancy_jsd_matrix(occupancy, x)
    write_metric_csv(
        jsd_df.reset_index().rename(columns={"index": "method"}), outfolder,
        "_".join(["cluster_occupancy_jsd", datatype, assembly]),
    )

    pooled_percentages = [pct for percentages in occupancy.values() for pct in percentages]
    bin_edges = compute_adaptive_occupancy_bins(pooled_percentages)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    fig = plt.figure(1, dpi=150, figsize=DEFAULT_FIGSIZE)
    ax = fig.subplots()

    # Grouped-bar histogram: within each bin, give every present method its
    # own thin bar side-by-side (rather than one line per method), so the
    # combined plot reads as a histogram like plot_cluster_occupancy_uplot_single.
    combos_to_plot = [combo for combo in x if combo in occupancy]
    n_methods = len(combos_to_plot)
    bin_widths = np.diff(bin_edges)
    bar_width = bin_widths / n_methods

    for i, combo in enumerate(combos_to_plot):
        percentages = occupancy[combo]
        counts, _ = np.histogram(percentages, bins=bin_edges)
        color = CONFIGDICT_COLOURS.get(combo)
        label = f"{FANCYDICT[combo]} (n={len(percentages)})"
        # offset this method's bars within each bin
        lefts = bin_edges[:-1] + i * bar_width
        ax.bar(
            lefts, counts, width=bar_width, align="edge",
            label=label, color=color, edgecolor="black", linewidth=0.3,
        )

    ax.set_xlabel(
        "isolates/strains represented in cluster (%)",
        fontsize=AXIS_TITLE_FONT_SIZE, fontproperties=ibmplexsans,
    )
    ax.set_ylabel(
        "number of clusters", fontsize=AXIS_TITLE_FONT_SIZE, fontproperties=ibmplexsans,
    )
    ax.set_xlim(left=0, right=100)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper left", fontsize=BASE_FONT_SIZE, frameon=True)
    ax.set_title(
        f"Cluster occupancy distribution (U plot) — {namedict[assembly]}{get_datatype_title_suffix(datatype)}",
        fontproperties=ibmplexsansbold,
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)
    fig.tight_layout()

    save_figure(fig, outfolder, "_".join(["plot_cluster_occupancy_uplot", datatype, assembly]))
    plt.close(fig)


def plot_cluster_occupancy_uplot_single(theargs):
    print(f"[TRACE] >>> Entering plot_cluster_occupancy_uplot_single() - defined at line 4421 of {__file__}")
    """Same cluster-occupancy distribution as plot_cluster_occupancy_uplot,
    but rendered as one bar-chart figure per method instead of everything
    overlaid on one set of axes -- much easier to read when you just want
    to look at a single method's U shape without the other curves getting
    in the way.

    Two differences from the overlaid version, both deliberate:
      1. Rendered as bars (one bar per adaptive bin) rather than a line,
         since a lone curve reads less clearly as a "distribution" than a
         histogram does.
      2. Bins are the same fixed-width 2-percentage-point edges as the
         overlaid plot (compute_adaptive_occupancy_bins), so a single
         method's histogram lines up exactly with its curve on the
         combined plot.

    One PNG/PDF/SVG file is written per method (filename includes the
    method name), in addition to the combined overlaid plot.
    """
    labels_by_seed, namedict, outfolder, assembly, datatype, font_props = theargs
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props

    print(f"\t- Plotting per-method cluster-occupancy U plots for {namedict[assembly]}")

    if not labels_by_seed:
        warnings.warn(
            f"No per-method label data available for {assembly}/{datatype}; "
            "skipping per-method cluster-occupancy U plots",
            RuntimeWarning, stacklevel=2,
        )
        return

    combos_present = set()
    for combo_dict in labels_by_seed.values():
        combos_present.update(combo_dict.keys())

    x = [combo for combo in COMBO_ORDER if combo in combos_present and combo in FANCYDICT]

    if not x:
        warnings.warn(
            f"No clusterer/sequence-type combos available for {assembly}/{datatype}; "
            "skipping per-method cluster-occupancy U plots",
            RuntimeWarning, stacklevel=2,
        )
        return

    total_strains = estimate_total_strains(labels_by_seed)
    if total_strains == 0:
        warnings.warn(
            f"Could not infer any strain identifiers for {assembly}/{datatype}; "
            "skipping per-method cluster-occupancy U plots",
            RuntimeWarning, stacklevel=2,
        )
        return

    occupancy = compute_occupancy_by_combo(labels_by_seed, x, total_strains)

    if not occupancy:
        warnings.warn(
            f"No cluster/strain data available for {assembly}/{datatype}; "
            "skipping per-method cluster-occupancy U plots",
            RuntimeWarning, stacklevel=2,
        )
        return

    # Safe filename stem per combo, e.g. "cdhit/nt" -> "cdhit_nt".
    combo_slug = lambda combo: combo.replace("/", "_")

    for combo in x:
        if combo not in occupancy:
            continue
        percentages = occupancy[combo]
        bin_edges = compute_adaptive_occupancy_bins(percentages)
        counts, _ = np.histogram(percentages, bins=bin_edges)
        widths = np.diff(bin_edges)
        color = CONFIGDICT_COLOURS.get(combo)

        fig = plt.figure(1, dpi=150, figsize=(7, 5))
        ax = fig.subplots()
        ax.bar(
            bin_edges[:-1], counts, width=widths, align="edge",
            color=color, edgecolor="black", linewidth=0.5,
        )

        ax.set_xlabel(
            "isolates/strains represented in cluster (%)",
            fontsize=AXIS_TITLE_FONT_SIZE, fontproperties=ibmplexsans,
        )
        ax.set_ylabel(
            "number of clusters", fontsize=AXIS_TITLE_FONT_SIZE, fontproperties=ibmplexsans,
        )
        ax.set_xlim(left=0, right=100)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(
            f"Cluster occupancy — {FANCYDICT[combo]}, {namedict[assembly]}{get_datatype_title_suffix(datatype)}, "
            f"n={len(percentages)}",
            fontproperties=ibmplexsansbold, fontsize=AXIS_TITLE_FONT_SIZE,
        )
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(ibmplexsans)
        fig.tight_layout()

        save_figure(
            fig, outfolder,
            "_".join(["plot_cluster_occupancy_uplot_single", datatype, assembly, combo_slug(combo)]),
        )
        plt.close(fig)


def get_realdata_reference_gene_set(real_data_run_dir):
    print(f"[TRACE] >>> Entering get_realdata_reference_gene_set() - defined at line 4532 of {__file__}")
    """Find a 'total genes' reference for real data, so gene-deletion
    percentages can be computed the same way as for simulations (see
    compute_gene_deletion_dataframe), even though there is no ground truth.

    Real biological data has no ground-truth gene universe, but Panaroo's
    gene_data.csv (column 4, the gene id) lists every gene Prokka annotated
    in the input assembly -- i.e. the full gene set *before* any clustering
    or filtering by any method. That makes it the best available stand-in
    for "the original number of genes" (n_original_genes) that
    compute_gene_deletion_dataframe expects, letting every method
    (including Panaroo itself) be checked against the same fixed
    reference. This is only available when a Panaroo folder was actually
    run for this dataset; returns None otherwise.

    Rows whose internal Panaroo id (column 3) contains "refound" are
    excluded: these are sequences Panaroo reconstructs itself to patch
    annotation gaps and were never fed into the clustering step, so they
    are not part of the "genes available to be clustered" universe and
    would otherwise make every method look like it deleted more than it
    actually did.
    """
    if not os.path.isdir(real_data_run_dir):
        return None
    for folder_name in os.listdir(real_data_run_dir):
        gene_data_path = os.path.join(real_data_run_dir, folder_name, "panaroo", "gene_data.csv")
        if os.path.isfile(gene_data_path):
            gene_data = pd.read_csv(gene_data_path, header=None, low_memory=False)
            is_refound = gene_data[2].astype(str).str.contains("refound")
            return set(gene_data.loc[~is_refound, 3])
    return None


def compute_gene_deletion_dataframe(labels_by_seed, total_genes_by_seed, assembly):
    print(f"[TRACE] >>> Entering compute_gene_deletion_dataframe() - defined at line 4565 of {__file__}")
    """Build a tidy dataframe of deleted-gene percentages.

    For every (seed, combo) pair present in `labels_by_seed`:

        deleted_pct = (n_original_genes - n_genes_kept) / n_original_genes * 100

    Assumptions:
      - `total_genes_by_seed[seed]` is the number of genes in the ORIGINAL
        dataset for that seed (n_original_genes from get_info_from_folder),
        i.e. the full reference gene set before any clustering/filtering.
      - `labels_by_seed[seed][combo]` is a dict of {gene_id: cluster_label}
        for genes the method actually assigned to a cluster (see
        get_labels_list_from_df); its length is therefore the number of
        genes RETAINED by that method for that seed. Genes with no cluster
        assignment at all are treated as deleted, matching the same
        assumption already used for the pairwise F1/ARI heatmaps.
      - Seeds/combos for which the original gene count is unknown (missing
        from total_genes_by_seed) are skipped with a warning, since a
        deletion percentage cannot be computed without a reference.

    Returns a DataFrame with columns: assembly, seed, clusterer, seqtype,
    combo, n_original_genes, n_genes_kept, n_genes_deleted, deleted_pct.
    """
    rows = []
    for seed, combo_dict in labels_by_seed.items():
        n_original = total_genes_by_seed.get(seed)
        if not n_original:
            warnings.warn(
                f"No original gene count available for {assembly}/{seed}; "
                "skipping deletion-percentage calculation for this seed",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        for combo, gene_dict in combo_dict.items():
            n_kept = len(gene_dict)
            n_deleted = n_original - n_kept
            deleted_pct = (n_deleted / n_original) * 100.0
            clusterer, seqtype = combo.split("/") if "/" in combo else (combo, "")
            rows.append(
                {
                    "assembly": assembly,
                    "seed": seed,
                    "clusterer": clusterer,
                    "seqtype": seqtype,
                    "combo": combo,
                    "n_original_genes": n_original,
                    "n_genes_kept": n_kept,
                    "n_genes_deleted": n_deleted,
                    "deleted_pct": deleted_pct,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "assembly", "seed", "clusterer", "seqtype", "combo",
            "n_original_genes", "n_genes_kept", "n_genes_deleted", "deleted_pct",
        ],
    )


def compute_gene_addition_dataframe(addition_by_seed, total_genes_by_seed, assembly):
    print(f"[TRACE] >>> Entering compute_gene_addition_dataframe() - defined at line 4628 of {__file__}")
    """Real-data-only companion to compute_gene_deletion_dataframe.

    For every (seed, combo) pair present in `addition_by_seed`:

        added_pct = n_refound / n_original_genes * 100

    Assumptions (mirrors compute_gene_deletion_dataframe):
      - `total_genes_by_seed[seed]` is the same reference gene-set size
        used for the deletion percentage (Panaroo's gene_data.csv, refound
        rows excluded -- see get_realdata_reference_gene_set), so deleted%
        and added% are directly comparable/stackable against the same
        denominator.
      - `addition_by_seed[seed][combo]` is the number of Panaroo "refound"
        gene ids seen in that combo's clustering output (see
        `filter_refound_genes` / the "n_refound" field returned by
        `_process_one_realdata_folder`). This is 0 (not missing) for every
        non-panaroo combo, so they still appear in the output with
        added_pct == 0.

    Returns a DataFrame with columns: assembly, seed, clusterer, seqtype,
    combo, n_original_genes, n_genes_added, added_pct.
    """
    rows = []
    for seed, combo_dict in addition_by_seed.items():
        n_original = total_genes_by_seed.get(seed)
        if not n_original:
            warnings.warn(
                f"No original gene count available for {assembly}/{seed}; "
                "skipping addition-percentage calculation for this seed",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        for combo, n_added in combo_dict.items():
            added_pct = (n_added / n_original) * 100.0
            clusterer, seqtype = combo.split("/") if "/" in combo else (combo, "")
            rows.append(
                {
                    "assembly": assembly,
                    "seed": seed,
                    "clusterer": clusterer,
                    "seqtype": seqtype,
                    "combo": combo,
                    "n_original_genes": n_original,
                    "n_genes_added": n_added,
                    "added_pct": added_pct,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "assembly", "seed", "clusterer", "seqtype", "combo",
            "n_original_genes", "n_genes_added", "added_pct",
        ],
    )


def plot_gene_deletion_boxplot(theargs):
    print(f"[TRACE] >>> Entering plot_gene_deletion_boxplot() - defined at line 4687 of {__file__}")
    """Boxplot of the % of genes deleted by each clusterer/seqtype combo,
    with the distribution taken across random seeds.

    X-axis: clusterer/seqtype combo (e.g. "Panaroo", "PanX (AA)")
    Y-axis: % of genes deleted relative to the original dataset
    Each box: distribution of deleted_pct across all seeds for that combo,
        i.e. the spread of the boxplot shows how consistently (narrow
        box) or variably (wide box, long whiskers) a method drops genes
        across different simulated replicates.
    Biological reading: a method with a high median deletion percentage
    is silently discarding a large fraction of the true simulated genes
    before they ever reach the clustering step -- this is a hidden cost
    that agreement metrics computed only on the genes actually retained
    (ARI, purity, etc.) cannot reveal, since a method can score
    "perfectly" on a small, easy, cherry-picked subset of genes while
    deleting everything harder to place. Compare against the deletion-
    penalised pairwise F1 heatmap (plot_pairwise_f1_heatmap) for a metric
    that folds this deletion cost back into the agreement score itself.
    """
    labels_by_seed, total_genes_by_seed, namedict, outfolder, assembly, datatype, font_props = theargs
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props

    print(f"\t- Plotting gene-deletion boxplot for simulations of {namedict[assembly]}")

    if not labels_by_seed:
        warnings.warn(
            f"No per-method label data available for {assembly}/{datatype}; "
            "skipping gene-deletion boxplot",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    deletion_df = compute_gene_deletion_dataframe(labels_by_seed, total_genes_by_seed, assembly)
    if deletion_df.empty:
        warnings.warn(
            f"No deletion-percentage data available for {assembly}/{datatype}; "
            "skipping gene-deletion boxplot",
            RuntimeWarning,
            stacklevel=2,
        )
        return deletion_df

    combos_present = set(deletion_df["combo"])
    x = [combo for combo in COMBO_ORDER if combo in combos_present and combo in FANCYDICT]
    if not x:
        warnings.warn(
            f"No clusterer/sequence-type combos available for {assembly}/{datatype}; "
            "skipping gene-deletion boxplot",
            RuntimeWarning,
            stacklevel=2,
        )
        return deletion_df

    x = [
        combo for combo in x
        if np.any(deletion_df.loc[deletion_df["combo"] == combo, "deleted_pct"].values > 0)
    ]
    if not x:
        warnings.warn(
            f"No methods with non-zero deleted genes for {assembly}/{datatype}; "
            "skipping gene-deletion boxplot",
            RuntimeWarning,
            stacklevel=2,
        )
        return deletion_df

    data = [deletion_df.loc[deletion_df["combo"] == combo, "deleted_pct"].values for combo in x]
    labels = [
        FANCYDICT[c] + (" *" if c.split("/")[0] in SKETCH_METHOD_NAMES or c.split("/")[0] in EMBED_METHOD_NAMES else "")
        for c in x
    ]

    fig = plt.figure(1, dpi=150, figsize=(max(6.0, len(x) * 0.6 + 2.0), 6.0))
    ax = fig.subplots()

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=True,
    )
    for patch, combo in zip(bp["boxes"], x):
        patch.set_facecolor(CONFIGDICT_COLOURS.get(combo, "#CCCCCC"))
        patch.set_alpha(0.8)

    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Genes deleted (%)", fontsize=AXIS_TITLE_FONT_SIZE, fontproperties=ibmplexsans)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_title(
        f"Gene deletion across seeds — {namedict[assembly]}{get_datatype_title_suffix(datatype)}",
        fontproperties=ibmplexsansbold,
    )
    fig.tight_layout()

    save_figure(fig, outfolder, f"plot_boxplot_gene_deletion_pct_{assembly}_{datatype}")
    plt.close(fig)

    return deletion_df


def plot_gene_deletion_and_addition_boxplot_realdata(theargs):
    print(f"[TRACE] >>> Entering plot_gene_deletion_and_addition_boxplot_realdata() - defined at line 4790 of {__file__}")
    """Real-data-only helper computing, for each clusterer/seqtype combo:
      A) % of genes DELETED relative to the original (Panaroo-annotated,
         refound-excluded) gene set -- reuses compute_gene_deletion_dataframe
         unchanged.
      B) % of genes ADDED by Panaroo (refound genes) relative to that same
         original gene set -- new, via compute_gene_addition_dataframe.
    Both percentages share the same denominator (n_original_genes), so
    they are directly comparable.

    === CHANGE (per user request) ===
    The boxplot figure this function used to produce for real data has
    been removed; only the two dataframes are still computed and
    returned here. main() still writes these out as the
    "gene_deletion_percentages_real_data.csv" and
    "gene_addition_percentages_real_data.csv" CSV files -- no PNG plot is
    generated for this any more. (Despite the function name being kept
    unchanged for minimal diff elsewhere in main()'s call sites.)

    This does not touch/replace `plot_gene_deletion_boxplot`, which is
    still used, unmodified, by the simulation branch.
    """
    (
        labels_by_seed, addition_by_seed, total_genes_by_seed,
        namedict, outfolder, assembly, datatype, font_props,
    ) = theargs

    print(f"\t- Computing gene deletion/addition percentages for {namedict[assembly]}")

    if not labels_by_seed:
        warnings.warn(
            f"No per-method label data available for {assembly}/{datatype}; "
            "skipping gene deletion/addition computation",
            RuntimeWarning, stacklevel=2,
        )
        return None, None

    deletion_df = compute_gene_deletion_dataframe(labels_by_seed, total_genes_by_seed, assembly)
    addition_df = compute_gene_addition_dataframe(addition_by_seed, total_genes_by_seed, assembly)

    if deletion_df.empty and addition_df.empty:
        warnings.warn(
            f"No deletion/addition data available for {assembly}/{datatype}",
            RuntimeWarning, stacklevel=2,
        )
        return deletion_df, addition_df

    return deletion_df, addition_df


def _plot_triangular_pairwise_heatmap(
    mat, x, labels, namedict, outfolder, assembly, datatype, font_props,
    cbar_label, filename_prefix,
):
    print(f"[TRACE] >>> Entering _plot_triangular_pairwise_heatmap() - defined at line 4840 of {__file__}")
    """Shared plotting code for the lower-triangle method-vs-method heatmaps
    (pairwise ARI, pairwise AMI/purity/V-measure, and pairwise gene-
    retention F1). `mat` is expected to already have its upper triangle
    (j > i) set to NaN by its caller's matrix-building function, since
    every one of these metrics is symmetric between methods i and j.

    Generic figure interpretation (shared by every caller of this
    function): rows and columns are both clustering methods, in the same
    order; only the lower triangle is drawn (upper triangle masked to
    white via `cmap.set_bad`, since it would just duplicate the lower
    triangle); the diagonal is always 1.0; cell shading follows the
    "YlGnBu" colourmap, scaled between the matrix's own min/max value, so
    darker/more saturated cells indicate stronger pairwise agreement (the
    exact metric being shown, and how to read its value, is described by
    `cbar_label` and by the specific calling function's own docstring).
    Numeric values are also printed directly inside each cell, with text
    colour flipped to white on dark cells for readability.

    Input:
        mat            -- (n x n) matrix to plot (lower triangle filled,
                           upper triangle NaN, diagonal 1.0).
        x              -- ordered list of combo keys (rows/columns).
        labels         -- display labels for each row/column (usually
                           FANCYDICT[...] with a sketch/embeddings marker).
        namedict, outfolder, assembly, datatype, font_props -- standard
                           plotting bookkeeping (see other plot functions).
        cbar_label     -- text label for the colourbar, describing which
                           metric this particular heatmap shows.
        filename_prefix -- prefix used to build the output file name.
    Output: none (saves PNG/PDF/SVG figures to `outfolder`).
    """
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props

    fig = plt.figure(
        1, dpi=150,
        figsize=(max(6.0, len(x) * 0.5 + 2.0), max(6.0, len(x) * 0.5 + 2.0)),
    )
    ax = fig.subplots()

    masked_mat = np.ma.masked_invalid(mat)
    cmap = copy.copy(plt.get_cmap("YlGnBu"))
    cmap.set_bad(color="white")
    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)

    im = ax.imshow(
        mat,
        cmap="YlGnBu",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(x)))
    ax.set_yticklabels(labels)

    for i in range(len(x)):
        for j in range(len(x)):
            val = mat[i, j]
            if np.isnan(val):
                continue
            txt_color = "white" if val < 0.5 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=BASE_FONT_SIZE, color=txt_color,
                fontproperties=ibmplexsans,
            )

    ax.set_xticks(np.arange(len(x) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(x) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", bottom=False, left=False)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(cbar_label, fontproperties=ibmplexsans)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)

    plt.text(0, 1.03, namedict[assembly], fontproperties=ibmplexsansitalics,
             horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)

    family_footnote = get_family_footnote(x)
    if family_footnote is not None:
        plt.text(
            0.5, -0.2, family_footnote,
            fontproperties=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1,
            horizontalalignment="center", verticalalignment="top", transform=ax.transAxes,
        )

    save_figure(fig, outfolder, "_".join([filename_prefix, datatype, assembly]), bbox_inches="tight")
    plt.close(fig)
    del fig, ax

    # requirement 6: export the exact matrix values behind this heatmap
    # (covers pairwise ARI/AMI/purity/V-measure/F1/exact-match, since they
    # all share this function) -- full precision, long format for easy
    # downstream use, plus the wide method x method matrix for reference.
    full_mat = np.where(np.isnan(mat), mat.T, mat)
    np.fill_diagonal(full_mat, 1.0)
    write_metric_csv(
        pd.DataFrame({
            "method_i": [labels[i] for i in range(len(x)) for j in range(len(x))],
            "method_j": [labels[j] for i in range(len(x)) for j in range(len(x))],
            "metric": cbar_label,
            "value": full_mat.flatten(),
        }),
        outfolder, "_".join([filename_prefix, datatype, assembly]),
    )

    # Every caller of this shared function passes a matrix that is
    # symmetric between methods i and j (upper triangle NaN'd out by the
    # caller purely for display; see docstring), so a dendrogram is
    # always meaningful here. The heatmap figure above must be closed
    # first, since both this and plot_nj_tree_from_matrix reuse figure
    # number 1.
    plot_nj_tree_from_matrix(
        mat, x, labels, namedict, outfolder, assembly, datatype, font_props,
        filename_prefix=filename_prefix + "_dendrogram",
        title_suffix=cbar_label,
    )


def plot_pairwise_ari_heatmap(theargs):
    print(f"[TRACE] >>> Entering plot_pairwise_ari_heatmap() - defined at line 4956 of {__file__}")
    """Draw the lower-triangle heatmap of pairwise inter-method Adjusted
    Rand Index (built by build_pairwise_ari_matrix).

    Figure interpretation:
        - Rows and columns are both clustering methods (clusterer/seqtype
          combos), in COMBO_ORDER; only the lower triangle is drawn (the
          metric is symmetric, so the upper triangle would be redundant
          and is masked to white).
        - Cell color/value = Adjusted Rand Index between that pair of
          methods' clusterings, computed on the genes both methods
          assigned to a cluster; darker/higher-value cells (colourbar,
          "YlGnBu") mean the two methods agree more strongly with each
          other on how genes should be grouped.
        - The diagonal is always 1.0 (a method trivially agrees with
          itself).
    Biological reading: methods that consistently show high ARI with
    every other method are giving a "consensus" pangenome structure;
    a method that is an outlier (low ARI vs. everything else) is
    clustering genes into families in a substantially different way from
    the rest, which is worth investigating (over/under-splitting, or a
    genuinely different notion of homology).

    Input/Output: same tuple shape as the other pairwise-heatmap
        functions; saves PNG/PDF/SVG via _plot_triangular_pairwise_heatmap.
    """
    labels_by_seed, namedict, outfolder, assembly, datatype, font_props = theargs
    ibmplexsans, ibmplexsansitalics, ibmplexsansbold = font_props

    print(f"\t- Plotting pairwise inter-method ARI heatmap for simulations of {namedict[assembly]}")

    if not labels_by_seed:
        warnings.warn(
            f"No per-method label data available for {assembly}/{datatype}; "
            "skipping pairwise ARI heatmap",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    combos_present = set()
    for combo_dict in labels_by_seed.values():
        combos_present.update(combo_dict.keys())

    x = [combo for combo in COMBO_ORDER if combo in combos_present and combo in FANCYDICT]

    if not x:
        warnings.warn(
            f"No clusterer/sequence-type combos available for {assembly}/{datatype}; "
            "skipping pairwise ARI heatmap",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    mat = build_pairwise_ari_matrix(x, labels_by_seed)

    labels = [
        FANCYDICT[c] + (" *" if c.split("/")[0] in SKETCH_METHOD_NAMES or c.split("/")[0] in EMBED_METHOD_NAMES else "")
        for c in x
    ]

    _plot_triangular_pairwise_heatmap(
        mat, x, labels, namedict, outfolder, assembly, datatype, font_props,
        cbar_label="Adjusted Rand index between methods (adim.)",
        filename_prefix="plot_heatmap_pairwise_ari",
    )


# Display name / colourbar label / output-filename info for each of the
# non-ARI pairwise agreement metrics, so plot_pairwise_metric_heatmap can
# stay generic (see build_pairwise_metric_matrix for the matching matrix
# builders, and plot_pairwise_ari_heatmap above for the ARI-specific
# version this mirrors).
PAIRWISE_METRIC_PLOT_INFO = {
    "ami": {
        "display_name": "AMI",
        "cbar_label": "Adjusted mutual information between methods (adim.)",
        "filename_prefix": "plot_heatmap_pairwise_ami",
    },
    "purity": {
        "display_name": "purity",
        "cbar_label": "Purity between methods (adim.)",
        "filename_prefix": "plot_heatmap_pairwise_purity",
    },
    "v_measure": {
        "display_name": "V-measure",
        "cbar_label": "V-measure between methods (adim.)",
        "filename_prefix": "plot_heatmap_pairwise_vmeasure",
    },
}


def plot_pairwise_metric_heatmap(theargs, metric_name):
    print(f"[TRACE] >>> Entering plot_pairwise_metric_heatmap() - defined at line 5049 of {__file__}")
    """Equivalent of plot_pairwise_ari_heatmap, but for one of the other
    method-vs-method agreement metrics (AMI, purity, V-measure); see
    PAIRWISE_METRIC_PLOT_INFO for the per-metric labelling/filename info,
    and build_pairwise_metric_matrix for how the underlying matrix is
    computed."""
    labels_by_seed, namedict, outfolder, assembly, datatype, font_props = theargs
    plot_info = PAIRWISE_METRIC_PLOT_INFO[metric_name]

    print(
        f"\t- Plotting pairwise inter-method {plot_info['display_name']} heatmap "
        f"for simulations of {namedict[assembly]}"
    )

    if not labels_by_seed:
        warnings.warn(
            f"No per-method label data available for {assembly}/{datatype}; "
            f"skipping pairwise {plot_info['display_name']} heatmap",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    combos_present = set()
    for combo_dict in labels_by_seed.values():
        combos_present.update(combo_dict.keys())

    x = [combo for combo in COMBO_ORDER if combo in combos_present and combo in FANCYDICT]

    if not x:
        warnings.warn(
            f"No clusterer/sequence-type combos available for {assembly}/{datatype}; "
            f"skipping pairwise {plot_info['display_name']} heatmap",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    mat = build_pairwise_metric_matrix(x, labels_by_seed, metric_name)

    labels = [
        FANCYDICT[c] + (" *" if c.split("/")[0] in SKETCH_METHOD_NAMES or c.split("/")[0] in EMBED_METHOD_NAMES else "")
        for c in x
    ]

    _plot_triangular_pairwise_heatmap(
        mat, x, labels, namedict, outfolder, assembly, datatype, font_props,
        cbar_label=plot_info["cbar_label"],
        filename_prefix=plot_info["filename_prefix"],
    )


def plot_pairwise_ami_heatmap(theargs):
    print(f"[TRACE] >>> Entering plot_pairwise_ami_heatmap() - defined at line 5101 of {__file__}")
    """Lower-triangle pairwise heatmap of inter-method Adjusted Mutual
    Information (see plot_pairwise_ari_heatmap for the shared figure
    layout/interpretation -- same rows/columns/masking/colourbar
    conventions, but cells hold AMI instead of ARI: an information-
    theoretic agreement score between two methods' clusterings, also in
    ~[0, 1] with 1 = identical clusterings)."""
    plot_pairwise_metric_heatmap(theargs, "ami")


def plot_pairwise_purity_heatmap(theargs):
    print(f"[TRACE] >>> Entering plot_pairwise_purity_heatmap() - defined at line 5111 of {__file__}")
    """Lower-triangle pairwise heatmap of inter-method purity (see
    plot_pairwise_ari_heatmap for the shared figure layout -- here each
    cell is the symmetrised purity score (_pairwise_purity_score) between
    two methods' clusterings, i.e. how much one method's clusters map
    onto single clusters of the other, averaged in both directions)."""
    plot_pairwise_metric_heatmap(theargs, "purity")


def plot_pairwise_vmeasure_heatmap(theargs):
    print(f"[TRACE] >>> Entering plot_pairwise_vmeasure_heatmap() - defined at line 5120 of {__file__}")
    """Lower-triangle pairwise heatmap of inter-method V-measure (see
    plot_pairwise_ari_heatmap for the shared figure layout -- here each
    cell is the V-measure, the harmonic mean of homogeneity and
    completeness, between two methods' clusterings)."""
    plot_pairwise_metric_heatmap(theargs, "v_measure")


def plot_pairwise_f1_heatmap(theargs):
    print(f"[TRACE] >>> Entering plot_pairwise_f1_heatmap() - defined at line 5128 of {__file__}")
    """Lower-triangle heatmap of the pairwise gene-retention F1 score
    between methods: how much agreement there is between two methods' sets
    of *kept* (non-deleted/non-filtered) genes, now penalised (see
    build_pairwise_f1_matrix) for genes deleted relative to the original
    dataset (requirement 1).

    Figure interpretation: same row/column/masking/colourbar conventions
    as plot_pairwise_ari_heatmap (see its docstring), but here each cell
    is the deletion-penalised Dice/F1 score between two methods' KEPT gene
    sets (build_pairwise_f1_matrix), not a clustering-structure agreement
    metric -- it answers "do these two methods keep (and thus implicitly
    also delete) the same genes?", independent of how those genes end up
    grouped into clusters.
    Biological reading: a low score between two methods can mean either
    that they disagree about which genes are real, or that one or both
    have deleted a large fraction of the original gene set (since the
    penalty term explicitly lowers the score as more of the fixed
    original gene count N goes unrecovered) -- pair this heatmap with
    plot_gene_deletion_boxplot to distinguish the two causes.
    """

    labels_by_seed, total_genes_by_seed, namedict, outfolder, assembly, datatype, font_props = theargs

    print(f"\t- Plotting pairwise gene-retention F1 heatmap for simulations of {namedict[assembly]}")

    if not labels_by_seed:
        warnings.warn(
            f"No per-method label data available for {assembly}/{datatype}; "
            "skipping pairwise gene-retention F1 heatmap",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    combos_present = set()
    for combo_dict in labels_by_seed.values():
        combos_present.update(combo_dict.keys())

    x = [combo for combo in COMBO_ORDER if combo in combos_present and combo in FANCYDICT]

    if not x:
        warnings.warn(
            f"No clusterer/sequence-type combos available for {assembly}/{datatype}; "
            "skipping pairwise gene-retention F1 heatmap",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    mat = build_pairwise_f1_matrix(x, labels_by_seed, total_genes_by_seed)

    labels = [
        FANCYDICT[c] + (" *" if c.split("/")[0] in SKETCH_METHOD_NAMES or c.split("/")[0] in EMBED_METHOD_NAMES else "")
        for c in x
    ]

    _plot_triangular_pairwise_heatmap(
        mat, x, labels, namedict, outfolder, assembly, datatype, font_props,
        # === CHANGE: label now reflects the gene-deletion penalty (requirement 1) ===
        cbar_label="Gene-retention adjusted Dice score between methods, penalised for deleted genes (adim.)",
        filename_prefix="plot_heatmap_pairwise_gene_retention_f1",
    )


def plot_pairwise_f1_heatmap_added_as_fp(theargs):
    print(f"[TRACE] >>> Entering plot_pairwise_f1_heatmap_added_as_fp() - defined at line 5193 of {__file__}")
    """Lower-triangle heatmap of the pairwise gene-retention F1 score
    between methods, using the "added_as_fp" formula (requirement 4):
    Panaroo-added/refound genes count as false positives, deleted genes
    count as false negatives against the original (Panaroo gene_data.csv,
    refound-excluded) reference gene set N.

        FP = a_i + a_j            (added/refound genes from either method)
        FN = N - TP               (TP = |kept_i ∩ kept_j|)
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN) = TP / N
        F1 = 2 * Precision * Recall / (Precision + Recall)

    Real-data only: called from main()'s real_data branch alongside (not
    instead of) plot_pairwise_f1_heatmap.
    """
    labels_by_seed, total_genes_by_seed, n_added_by_seed, namedict, outfolder, assembly, datatype, font_props = theargs

    print(f"\t- Plotting pairwise gene-retention F1 (added-as-FP) heatmap for {namedict[assembly]}")

    if not labels_by_seed:
        warnings.warn(
            f"No per-method label data available for {assembly}/{datatype}; "
            "skipping pairwise gene-retention F1 (added-as-FP) heatmap",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    combos_present = set()
    for combo_dict in labels_by_seed.values():
        combos_present.update(combo_dict.keys())

    x = [combo for combo in COMBO_ORDER if combo in combos_present and combo in FANCYDICT]

    if not x:
        warnings.warn(
            f"No clusterer/sequence-type combos available for {assembly}/{datatype}; "
            "skipping pairwise gene-retention F1 (added-as-FP) heatmap",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    mat = build_pairwise_f1_matrix_with_additions(
        x, labels_by_seed, total_genes_by_seed, n_added_by_seed, mode="added_as_fp",
    )

    labels = [
        FANCYDICT[c] + (" *" if c.split("/")[0] in SKETCH_METHOD_NAMES or c.split("/")[0] in EMBED_METHOD_NAMES else "")
        for c in x
    ]

    _plot_triangular_pairwise_heatmap(
        mat, x, labels, namedict, outfolder, assembly, datatype, font_props,
        cbar_label=(
            "Gene-retention adjusted Dice score between methods, penalised "
            "for deleted AND Panaroo-added/refound genes (adim.)"
        ),
        filename_prefix="plot_heatmap_pairwise_gene_retention_f1_added_as_fp",
    )


def load_seeds(seedsfile):
    print(f"[TRACE] >>> Entering load_seeds() - defined at line 5256 of {__file__}")
    """Read the list of random-seed integers to analyse (one per line)
    from a plain-text file, sorted ascending.

    Input:  seedsfile -- path to a text file with one integer seed per
            (non-blank) line.
    Output: sorted list of int seeds, later used to iterate over every
    simulation replicate for each assembly.
    """
    seeds = []
    with open(seedsfile, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                seeds.append(int(stripped))
    seeds.sort()
    return seeds


def build_results_dataframe(listoflists):
    print(f"[TRACE] >>> Entering build_results_dataframe() - defined at line 5275 of {__file__}")
    """Assemble the flat list of per-(assembly, seed, clusterer) result
    rows collected by get_info_from_folder / get_info_from_folder_realdata
    into the single indexed DataFrame (`outdf`) used by every downstream
    plotting/analysis function in this script.

    Input:  listoflists -- list of rows, each matching the column order
            below (simulations flag, assembly, seed, clusterer, the
            truth-agreement metrics [NaN for real data], cluster counts,
            clustering parameters in PARAMORDER, and runtime).
    Output: a DataFrame with columns
        [adj_rand_index, adj_rand_index_pvalue, purity, adj_mutual_info,
         adj_mutual_info_pvalue, homogeneity, completeness, v_measure,
         n_clusters, n_singletons, n_pairs]
        + PARAMORDER + [runtime], indexed by a MultiIndex of
        (simulations, assembly, seed, clusterer) so rows can be sliced by
        any combination of those four keys elsewhere in the script.

        adj_rand_index_pvalue and adj_mutual_info_pvalue are empirical
        permutation-test p-values (see permutation_test_agreement) for
        the ARI/AMI columns immediately preceding them -- NaN for real
        data (no ground truth to test against), same as the other
        truth-agreement columns. No p-value column is included for
        purity/homogeneity/completeness/v_measure (see
        calculate_values_from_cluster_matrix's docstring for why).
    """
    outdf = pd.DataFrame(
        listoflists,
        columns=[
            "simulations",
            "assembly",
            "seed",
            "clusterer",
            "adj_rand_index",
            ADJ_RAND_INDEX_PVALUE_COL,
            "purity",
            "adj_mutual_info",
            ADJ_MUTUAL_INFO_PVALUE_COL,
            "homogeneity",
            "completeness",
            "v_measure",
            "n_clusters",
            "n_singletons",
            "n_pairs",
        ]
        + PARAMORDER
        + ["runtime"],
    )
    index = pd.MultiIndex.from_frame(outdf[["simulations", "assembly", "seed", "clusterer"]])
    outdf = outdf.drop(["simulations", "assembly", "seed", "clusterer"], axis=1)
    return outdf.set_index(index)


def discover_analysis_tasks(runfolder, datapath, seeds):
    print(f"[TRACE] >>> Entering discover_analysis_tasks() - defined at line 5325 of {__file__}")
    """Walk the SIMULATION results directory tree and build the list of
    (assembly, seed) work units to analyse, verifying that both the
    clustering-tool result folder AND the matching ground-truth folder
    exist for each one before scheduling it.

    Input:
        runfolder -- root directory containing the "simulations"
                     subfolder of clustering-tool outputs.
        datapath  -- root directory containing the matching
                     "simulations" subfolder of ground-truth data.
        seeds     -- list of seed integers to look for under every
                     discovered assembly (see load_seeds).
    Output: (tasks, missing) -- `tasks` is a list of
        (simulations_run_dir, assembly, seed, datapath) tuples ready to
        be passed to get_info_from_folder; `missing` lists any
        (assembly, seed) combination where either the result or the
        ground-truth folder was absent, together with the paths that
        were checked, so report_missing_tasks can explain the gap.
    """
    simulations_run_dir = os.path.join(runfolder, "simulations")
    tasks = []
    missing = []

    for assembly in next(os.walk(simulations_run_dir))[1]:
        for seed in seeds:
            result_seed_dir = os.path.join(simulations_run_dir, assembly, str(seed))
            truth_seed_dir = os.path.join(datapath, "simulations", assembly, str(seed))

            if not os.path.isdir(result_seed_dir) or not os.path.isdir(truth_seed_dir):
                missing.append((assembly, seed, result_seed_dir, truth_seed_dir))
                continue

            tasks.append((simulations_run_dir, assembly, seed, datapath))

    return tasks, missing


def report_missing_tasks(missingtasks, gettinginfotasks):
    print(f"[TRACE] >>> Entering report_missing_tasks() - defined at line 5363 of {__file__}")
    """Print a human-readable warning listing every simulation
    (assembly, seed) combination discover_analysis_tasks could not find a
    complete result+ground-truth folder pair for, so a partial run
    doesn't silently proceed unnoticed.

    Input:
        missingtasks     -- list of (assembly, seed, result_seed_dir,
                             truth_seed_dir) tuples from
                             discover_analysis_tasks.
        gettinginfotasks -- list of tasks that WERE found (used only to
                             report how many are being analysed anyway).
    Output: none (prints/warns only); no-op if missingtasks is empty.
    """
    if not missingtasks:
        return

    warnings.warn(
        f"{len(missingtasks)} expected simulations are missing; "
        f"analysing {len(gettinginfotasks)} present simulations",
        RuntimeWarning,
        stacklevel=2,
    )
    print("> Missing expected simulations:")
    for assembly, seed, result_seed_dir, truth_seed_dir in missingtasks:
        print(f"\t- {assembly}/{seed}")
        if not os.path.isdir(result_seed_dir):
            print(f"\t  missing result folder: {result_seed_dir}")
        if not os.path.isdir(truth_seed_dir):
            print(f"\t  missing truth folder: {truth_seed_dir}")


def discover_analysis_tasks_realdata(runfolder, seeds):
    print(f"[TRACE] >>> Entering discover_analysis_tasks_realdata() - defined at line 5395 of {__file__}")
    """Real-data equivalent of discover_analysis_tasks.

    === CHANGE: flat layout, no assembly/seed subdirectories ===========
    Your actual real-data layout is flat: runfolder/real_data/<clusterer_
    paramdir>/... directly, with no per-assembly or per-seed nesting (no
    more "seeds" for real data -- there is one run). This is different from
    the nested runfolder/simulations/<assembly>/<seed>/... layout, so this
    function does not mirror discover_analysis_tasks's directory walk.

    The `seeds` argument is accepted only to keep main()'s call signature
    symmetric with the simulations branch; it is ignored here, since real
    data has no seed dimension to iterate over. There is also no
    ground-truth directory to check for (unlike discover_analysis_tasks),
    since real data has none.
    """
    real_data_run_dir = os.path.join(runfolder, "real_data")
    tasks = []
    missing = []

    if not os.path.isdir(real_data_run_dir):
        missing.append(("real_data", "run", real_data_run_dir))
        return tasks, missing

    # Single pseudo-assembly ("real_data") / pseudo-seed ("run") -- see the
    # docstring of get_info_from_folder_realdata for why. datapath (4th
    # element) is unused for real data, kept as None purely so the task
    # tuple shape matches the simulations path structurally.
    tasks.append((real_data_run_dir, "real_data", "run", None))

    return tasks, missing


def report_missing_tasksrealdata(missingtasks, gettinginfotasks):
    print(f"[TRACE] >>> Entering report_missing_tasksrealdata() - defined at line 5428 of {__file__}")
    """Real-data equivalent of report_missing_tasks: warns if the expected
    "real_data" result directory itself is missing (there's only ever one
    such pseudo-task in real-data mode, see discover_analysis_tasks_realdata).

    Input/Output: same shape/behaviour as report_missing_tasks, but for
        the single flat real-data run directory rather than a per-
        (assembly, seed) simulation grid.
    """
    if not missingtasks:
        return

    warnings.warn(
        f"{len(missingtasks)} expected real-data result folders are missing; "
        f"analysing {len(gettinginfotasks)} present ones",
        RuntimeWarning,
        stacklevel=2,
    )
    print("> Missing expected real-data result folders:")
    for assembly, seed, result_seed_dir in missingtasks:
        print(f"\t- {assembly}/{seed}: missing result folder: {result_seed_dir}")


def main():
    print(f"[TRACE] >>> Entering main() - defined at line 5451 of {__file__}")
    """Command-line entry point for the whole clustering-benchmark
    analysis pipeline.

    High-level flow:
      1. Parse CLI arguments (data/run folder paths, font paths, number
         of parallel workers, and --real-data to switch pipelines).
      2. SIMULATION mode (default): for every simulated assembly and
         random seed, load each clustering tool's output and the known
         ground truth, compute agreement metrics (ARI/purity/AMI/
         V-measure/p-value) via get_info_from_folder, and generate the
         full suite of comparison plots (c-sweeps, point plots, violin/
         stacked-bar cluster-count plots, method-vs-method heatmaps,
         gene-deletion boxplots) plus a combined results CSV.
      3. REAL-DATA mode (--real-data): for the single flat real-data run
         directory, load only the ground-truth-free clusterers (CD-HIT,
         MMseqs2, DIAMOND, Panaroo) via get_info_from_folder_realdata,
         and generate the subset of plots that don't require a known
         truth (runtime, cluster counts, method-vs-method agreement
         heatmaps including the new exact-cluster-match heatmap, the new
         core-genome estimation curve, and gene-deletion/-addition CSVs),
         since there is no simulated ground truth to score against here.
    Output: none directly returned; writes PNG/PDF/SVG figures and CSV/
        TXT summary tables into the output folder specified on the
        command line, and prints progress messages throughout.
    """
    parser = argparse.ArgumentParser(description="Analyse gene clustering benchmark runs.")
    parser.add_argument("runfolder", default="./")
    # "simulations" (default) preserves the exact original behaviour end to
    # end. "real_data" switches to the ground-truth-free pipeline: no
    # truth-comparison metrics, no gene-id-format assumptions.
    parser.add_argument(
        "--mode", choices=["simulations", "real_data"], default="simulations",
        help="'simulations' (default) analyses simulated runs against ground "
             "truth, exactly as before. 'real_data' analyses real biological "
             "data (folder 'real_data' instead of 'simulations'): no ground "
             "truth, no assumptions about gene ID format, and only "
             "CD-HIT/DIAMOND/MMseqs2 are considered.",
    )
    parser.add_argument("--out-folder", dest="outfolder", default="./temp_runanalysis")
    parser.add_argument("--nthreads", "-j", type=int, default=1)
    parser.add_argument("--datapath", default=DEFAULT_DATAPATH)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--font-regular", default=DEFAULT_FONT_REGULAR)
    parser.add_argument("--font-italic", default=DEFAULT_FONT_ITALIC)
    parser.add_argument("--font-bold", default=DEFAULT_FONT_BOLD)
    args = parser.parse_args()

    plt.rcParams.update({"figure.max_open_warning": 0, "font.size": BASE_FONT_SIZE})
    font_props = get_font_properties(args)
    seedsfile = args.seeds

    # === NEW: fail with a clear message instead of a bare StopIteration
    # from os.walk() when runfolder doesn't exist (e.g. a missing leading
    # "/" turning an absolute path into a relative one) ===
    if not os.path.isdir(args.runfolder):
        raise RuntimeError(
            f"runfolder does not exist or is not a directory: {args.runfolder!r} "
            f"(resolved to {os.path.abspath(args.runfolder)!r}). "
            "Check the path, including a leading '/' if it's meant to be absolute."
        )
    lsdirs = next(os.walk(args.runfolder))[1]

    if args.mode == "simulations":
        if "simulations" not in lsdirs:
            raise RuntimeError("No valid folders found!")
    else:
        if "real_data" not in lsdirs:
            raise RuntimeError(
                "No valid folders found! Expected a 'real_data' directory "
                f"under {args.runfolder} when --mode real_data is used."
            )

    print("> Getting seeds")
    seeds = load_seeds(seedsfile)
    print("> Got {} seeds".format(len(seeds)))

    print("\n> Getting info...")
    if args.mode == "simulations":
        gettinginfotasks, missingtasks = discover_analysis_tasks(
            args.runfolder,
            args.datapath,
            seeds,
        )
        report_missing_tasks(missingtasks, gettinginfotasks)
        if not gettinginfotasks:
            expected_dir = os.path.join(args.runfolder, "simulations")
            raise RuntimeError(
                "No analysable simulations found; expected result folders under "
                f"{expected_dir}/<assembly>/<seed>"
            )
    else:
        gettinginfotasks, missingtasks = discover_analysis_tasks_realdata(
            args.runfolder,
            seeds,
        )
        report_missing_tasksrealdata(missingtasks, gettinginfotasks)
        if not gettinginfotasks:
            expected_dir = os.path.join(args.runfolder, "real_data")
            raise RuntimeError(
                "No analysable real-data results found; expected result folders under "
                f"{expected_dir}/<assembly>/<seed>"
            )

    listoflists = []
    namedict = {}
    # assembly -> {seed: {combo: {gene_id: label}}}, used for the pairwise
    # inter-method ARI heatmap.
    method_labels_by_assembly = {}
    # Total number of genes in the ORIGINAL dataset for each assembly/seed,
    # used as the reference/universe for the gene-deletion penalty in the
    # pairwise F1 heatmap and for the per-method deletion-percentage boxplot.
    total_genes_by_assembly = {}
    # === (parallelization): real-data mode has only ONE outer task
    # (see discover_analysis_tasks_realdata), so routing it through main()'s
    # outer Pool.map would waste `-j` on a list of length 1. Instead we call
    # get_info_from_folder_realdata directly and let IT parallelize
    # internally, across clusterer folders, using args.nthreads. The
    # simulations path is untouched: it still has one outer task per
    # assembly/seed and benefits from the outer Pool exactly as before.

    addition_by_assembly = {}
    if args.mode == "real_data":
        for task in gettinginfotasks:
            tmpout = get_info_from_folder_realdata(task, nthreads=args.nthreads)
            listoflists += tmpout[0]
            namedict[tmpout[1]] = tmpout[2]
            method_labels_by_assembly.setdefault(tmpout[1], {})[tmpout[3]] = tmpout[4]
            total_genes_by_assembly.setdefault(tmpout[1], {})[tmpout[3]] = tmpout[5]
            addition_by_assembly.setdefault(tmpout[1], {})[tmpout[3]] = tmpout[6]
        print("\n> All real data retrieved, now waiting on plots...")
    elif args.nthreads <= 1:
        for task in gettinginfotasks:
            tmpout = get_info_from_folder(task)
            listoflists += tmpout[0]
            namedict[tmpout[1]] = tmpout[2]
            method_labels_by_assembly.setdefault(tmpout[1], {})[tmpout[3]] = tmpout[4]
            total_genes_by_assembly.setdefault(tmpout[1], {})[tmpout[3]] = tmpout[5]
    else:
        pool = Pool(args.nthreads)
        for result in pool.map(get_info_from_folder, gettinginfotasks):
            listoflists += result[0]
            namedict[result[1]] = result[2]
            method_labels_by_assembly.setdefault(result[1], {})[result[3]] = result[4]
            total_genes_by_assembly.setdefault(result[1], {})[result[3]] = result[5]
        pool.close()
        pool.join()
    print("\n> Done!")

    if not listoflists:
        raise RuntimeError("No valid clustering results were produced from the analysable " + args.mode)

    outdf = build_results_dataframe(listoflists)
    print("Clusterers found:", set(outdf.index.get_level_values("clusterer")))
    
    assemblies = set(list(outdf.index.get_level_values("assembly")))

    if not os.path.isdir(args.outfolder):
        os.makedirs(args.outfolder)

    datatype = args.mode  # "simulations" or "real_data"; used for filenames/labels below

    if args.mode == "simulations":
        # ================= ORIGINAL SIMULATIONS PIPELINE =================
        plotstodo = ["adj_rand_index", "purity", "adj_mutual_info", "v_measure", "runtime"]
        print("\n> Preparing plotting tasks...")
        plottingtasks_pointplots = [
            (plot_name, outdf, namedict, args.outfolder, assembly, datatype, font_props)
            for plot_name in plotstodo
            for assembly in assemblies
        ]
        plottingtasks = [
            (plot_name, outdf, namedict, args.outfolder, assembly, datatype, font_props)
            for plot_name in plotstodo
            for assembly in assemblies
        ]

        # Ground-truth number-of-clusters stats (mean/min/max across seeds)
        # per assembly, reusing get_truth_matrix_path/original_gene logic
        # from fast_count_groundtruth_clusters.sh -- used only to draw the
        # thin overlay on the simulation "n_clusters" violin plot below.
        print("\n> Computing ground-truth cluster-count stats per assembly "
              "(for the n_clusters violin-plot overlay)...")
        ground_truth_stats_by_assembly = {
            assembly: get_simulation_ground_truth_cluster_stats(args.datapath, assembly)
            for assembly in assemblies
        }

        plottingtasks_violin = [
            (
                "n_clusters", outdf, namedict, args.outfolder, assembly, datatype, font_props,
                ground_truth_stats_by_assembly.get(assembly),
            )
            for assembly in assemblies
        ]

        plottingtasks_stackedbar_c = [
            ("n_clusters", outdf, namedict, args.outfolder, assembly, datatype, font_props)
            for assembly in assemblies
        ]

        plottingtasks_stackedbar = [
            ("n_clusters", outdf, namedict, args.outfolder, assembly, datatype, font_props)
            for assembly in assemblies
        ]

        plottingtasks_heatmap = [
            ("method_comparison", outdf, namedict, args.outfolder, assembly, datatype, font_props)
            for assembly in assemblies
        ]

        plottingtasks_pairwise_ari = [
            (method_labels_by_assembly.get(assembly, {}), namedict, args.outfolder, assembly, datatype, font_props)
            for assembly in assemblies
        ]

        plottingtasks_pairwise_f1 = [
            (
                method_labels_by_assembly.get(assembly, {}),
                total_genes_by_assembly.get(assembly, {}),
                namedict, args.outfolder, assembly, datatype, font_props,
            )
            for assembly in assemblies
        ]

        plottingtasks_gene_deletion = [
            (
                method_labels_by_assembly.get(assembly, {}),
                total_genes_by_assembly.get(assembly, {}),
                namedict, args.outfolder, assembly, datatype, font_props,
            )
            for assembly in assemblies
        ]
        print("\n> Plotting...")

        deletion_dfs = []
        if args.nthreads <= 1:
            for task in plottingtasks_pointplots:
                plotter_pointplots(task)
            for task in plottingtasks:
                plotter(task)
            for task in plottingtasks_violin:
                number_of_clusters_violin(task)
            for task in plottingtasks_stackedbar:
                number_of_clusters_stacked_bar(task)
            for task in plottingtasks_stackedbar_c:
                number_of_clusters_stacked_bar_vs_c(task)
            for task in plottingtasks_heatmap:
                methods_comparison_heatmap(task)
            for task in plottingtasks_pairwise_ari:
                plot_pairwise_ari_heatmap(task)
                plot_pairwise_ami_heatmap(task)
                plot_pairwise_purity_heatmap(task)
                plot_pairwise_vmeasure_heatmap(task)
            for task in plottingtasks_pairwise_f1:
                plot_pairwise_f1_heatmap(task)
            for task in plottingtasks_gene_deletion:
                ddf = plot_gene_deletion_boxplot(task)
                if ddf is not None and not ddf.empty:
                    deletion_dfs.append(ddf)

        else:
            pool = Pool(args.nthreads)
            pool.map(plotter_pointplots, plottingtasks_pointplots)
            pool.map(plotter, plottingtasks)
            pool.map(number_of_clusters_violin, plottingtasks_violin)
            pool.map(number_of_clusters_stacked_bar, plottingtasks_stackedbar)
            pool.map(number_of_clusters_stacked_bar_vs_c, plottingtasks_stackedbar_c)
            pool.map(methods_comparison_heatmap, plottingtasks_heatmap)
            pool.map(plot_pairwise_ari_heatmap, plottingtasks_pairwise_ari)
            pool.map(plot_pairwise_ami_heatmap, plottingtasks_pairwise_ari)
            pool.map(plot_pairwise_purity_heatmap, plottingtasks_pairwise_ari)
            pool.map(plot_pairwise_vmeasure_heatmap, plottingtasks_pairwise_ari)
            pool.map(plot_pairwise_f1_heatmap, plottingtasks_pairwise_f1)
            for ddf in pool.map(plot_gene_deletion_boxplot, plottingtasks_gene_deletion):
                if ddf is not None and not ddf.empty:
                    deletion_dfs.append(ddf)
            pool.close()
            pool.join()

        outdf.to_csv(
            os.path.join(
                args.outfolder,
                "clustering_metrics.txt"
            ),
            sep="\t"
        )


        if deletion_dfs:
            combined_deletion_df = pd.concat(deletion_dfs, ignore_index=True)
            combined_deletion_df.to_csv(
                os.path.join(args.outfolder, "gene_deletion_percentages.csv"),
                index=False,
            )

    else:
        # ================= NEW: REAL-DATA PIPELINE =================

        # What is intentionally NOT run in this branch, and why:
        #   - plotter / plotter_pointplots for adj_rand_index / purity /
        #     adj_mutual_info / v_measure: these are agreement-with-truth
        #     metrics; every value for real data is NaN (no truth), so the
        #     plots would be empty/meaningless. Only "runtime" is kept.
        #   - methods_comparison_heatmap: its columns are exactly those same
        #     truth-based metrics, so it is skipped entirely.
        #   - plot_pairwise_f1_heatmap uses the *un-penalised* pairwise
        #     F1/Dice score between methods (build_pairwise_f1_matrix's
        #     documented fallback when total_genes_by_seed is not supplied),
        #     since that heatmap compares methods' kept-gene sets against
        #     EACH OTHER, not against a fixed reference -- a perfect 1.0
        #     there only means two methods kept the same genes as each
        #     other, not that neither of them deleted anything.
        #   - The per-method gene-deletion boxplot below answers that other
        #     question directly (how many genes did THIS method delete,
        #     relative to the full annotated gene set), using Panaroo's
        #     gene_data.csv as the reference universe when available (see
        #     get_realdata_reference_gene_set).

        print("\n> Preparing plotting tasks (real-data mode: runtime, cluster-count "
              "and method-vs-method comparisons only)...")

        plottingtasks_pointplots = [
            ("runtime", outdf, namedict, args.outfolder, assembly, datatype, font_props)
            for assembly in assemblies
        ]
        plottingtasks = [
            ("runtime", outdf, namedict, args.outfolder, assembly, datatype, font_props)
            for assembly in assemblies
        ]

        # Real data has no simulated ground truth, so the violin-plot
        # overlay is always disabled here (gt_stats=None); the plot is
        # rendered exactly as before.
        plottingtasks_violin = [
            ("n_clusters", outdf, namedict, args.outfolder, assembly, datatype, font_props, None)
            for assembly in assemblies
        ]

        plottingtasks_stackedbar_c = [
            ("n_clusters", outdf, namedict, args.outfolder, assembly, datatype, font_props)
            for assembly in assemblies
        ]

        plottingtasks_stackedbar = [
            ("n_clusters", outdf, namedict, args.outfolder, assembly, datatype, font_props)
            for assembly in assemblies
        ]

        plottingtasks_pairwise_ari = [
            (method_labels_by_assembly.get(assembly, {}), namedict, args.outfolder, assembly, datatype, font_props)
            for assembly in assemblies
        ]

        plottingtasks_exact_match = [
            (method_labels_by_assembly.get(assembly, {}), namedict, args.outfolder, assembly, datatype, font_props)
            for assembly in assemblies
        ]
        plottingtasks_core_genome = [
            (method_labels_by_assembly.get(assembly, {}), namedict, args.outfolder, assembly, datatype, font_props)
            for assembly in assemblies
        ]

        plottingtasks_uplot = [
            (method_labels_by_assembly.get(assembly, {}), namedict, args.outfolder, assembly, datatype, font_props)
            for assembly in assemblies
        ]

        plottingtasks_pairwise_f1 = [
            (
                method_labels_by_assembly.get(assembly, {}),
                None,
                namedict, args.outfolder, assembly, datatype, font_props,
            )
            for assembly in assemblies
        ]

        real_data_run_dir = gettinginfotasks[0][0]
        reference_genes = get_realdata_reference_gene_set(real_data_run_dir)
        if reference_genes is None:
            warnings.warn(
                "No Panaroo gene_data.csv found under "
                f"{real_data_run_dir}; skipping the per-method gene-deletion/"
                "addition check (no reference gene set available for real "
                "data without a Panaroo run)",
                RuntimeWarning,
                stacklevel=2,
            )
            plottingtasks_gene_deletion = []
            plottingtasks_pairwise_f1_added = []
        else:
            total_genes_by_seed_realdata = {"run": len(reference_genes)}
            plottingtasks_gene_deletion = [
                (
                    method_labels_by_assembly.get(assembly, {}),
                    addition_by_assembly.get(assembly, {}),
                    total_genes_by_seed_realdata,
                    namedict, args.outfolder, assembly, datatype, font_props,
                )
                for assembly in assemblies
            ]

            plottingtasks_pairwise_f1_added = [
                (
                    method_labels_by_assembly.get(assembly, {}),
                    total_genes_by_seed_realdata,
                    addition_by_assembly.get(assembly, {}),
                    namedict, args.outfolder, assembly, datatype, font_props,
                )
                for assembly in assemblies
            ]

        print("\n> Plotting...")
        deletion_dfs = []
        addition_dfs = []
        if args.nthreads <= 1:
            for task in plottingtasks_pointplots:
                plotter_pointplots(task)
            for task in plottingtasks:
                plotter(task)
            for task in plottingtasks_violin:
                number_of_clusters_violin(task)
            for task in plottingtasks_stackedbar:
                number_of_clusters_stacked_bar(task)
            for task in plottingtasks_stackedbar_c:
                number_of_clusters_stacked_bar_vs_c(task)
            for task in plottingtasks_pairwise_ari:
                plot_pairwise_ari_heatmap(task)
                plot_pairwise_ami_heatmap(task)
                plot_pairwise_purity_heatmap(task)
                plot_pairwise_vmeasure_heatmap(task)
            for task in plottingtasks_exact_match:
                plot_pairwise_exact_match_heatmap(task)
            for task in plottingtasks_core_genome:
                plot_core_genome_curve_realdata(task)
            for task in plottingtasks_uplot:
                plot_cluster_occupancy_uplot(task)
                plot_cluster_occupancy_uplot_single(task)
            for task in plottingtasks_pairwise_f1:
                plot_pairwise_f1_heatmap(task)
            for task in plottingtasks_pairwise_f1_added:
                plot_pairwise_f1_heatmap_added_as_fp(task)
            for task in plottingtasks_gene_deletion:
                ddf, adf = plot_gene_deletion_and_addition_boxplot_realdata(task)
                if ddf is not None and not ddf.empty:
                    deletion_dfs.append(ddf)
                if adf is not None and not adf.empty:
                    addition_dfs.append(adf)
        else:
            pool = Pool(args.nthreads)
            pool.map(plotter_pointplots, plottingtasks_pointplots)
            pool.map(plotter, plottingtasks)
            pool.map(number_of_clusters_stacked_bar, plottingtasks_stackedbar)
            pool.map(number_of_clusters_stacked_bar_vs_c, plottingtasks_stackedbar_c)
            pool.map(plot_pairwise_ari_heatmap, plottingtasks_pairwise_ari)
            pool.map(plot_pairwise_ami_heatmap, plottingtasks_pairwise_ari)
            pool.map(plot_pairwise_purity_heatmap, plottingtasks_pairwise_ari)
            pool.map(plot_pairwise_vmeasure_heatmap, plottingtasks_pairwise_ari)
            pool.map(plot_pairwise_exact_match_heatmap, plottingtasks_exact_match)
            pool.map(plot_core_genome_curve_realdata, plottingtasks_core_genome)
            pool.map(plot_cluster_occupancy_uplot, plottingtasks_uplot)
            pool.map(plot_cluster_occupancy_uplot_single, plottingtasks_uplot)
            pool.map(plot_pairwise_f1_heatmap, plottingtasks_pairwise_f1)
            if plottingtasks_pairwise_f1_added:
                pool.map(plot_pairwise_f1_heatmap_added_as_fp, plottingtasks_pairwise_f1_added)
            if plottingtasks_gene_deletion:
                for ddf, adf in pool.map(plot_gene_deletion_and_addition_boxplot_realdata, plottingtasks_gene_deletion):
                    if ddf is not None and not ddf.empty:
                        deletion_dfs.append(ddf)
                    if adf is not None and not adf.empty:
                        addition_dfs.append(adf)
            pool.close()
            pool.join()


        outdf.to_csv(
            os.path.join(args.outfolder, "clustering_metrics_real_data.txt"),
            sep="\t"
        )

        if deletion_dfs:
            combined_deletion_df = pd.concat(deletion_dfs, ignore_index=True)
            print("\n> Per-method gene deletion (relative to the full annotated gene set):")
            print(
                combined_deletion_df[["clusterer", "seqtype", "n_original_genes", "n_genes_kept", "n_genes_deleted", "deleted_pct"]]
                .to_string(index=False)
            )
            combined_deletion_df.to_csv(
                os.path.join(args.outfolder, "gene_deletion_percentages_real_data.csv"),
                index=False,
            )

        if addition_dfs:
            combined_addition_df = pd.concat(addition_dfs, ignore_index=True)
            print("\n> Per-method gene addition (Panaroo refound genes, relative to the full annotated gene set):")
            print(
                combined_addition_df[["clusterer", "seqtype", "n_original_genes", "n_genes_added", "added_pct"]]
                .to_string(index=False)
            )
            combined_addition_df.to_csv(
                os.path.join(args.outfolder, "gene_addition_percentages_real_data.csv"),
                index=False,
            )

    print("\n> Done!\n")


if __name__ == "__main__":
    main()
