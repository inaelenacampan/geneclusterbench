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

# keep in mind to install the needed packages for fonts

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
# === NEW (real-data support) ===
# Real biological data has no ground truth and gene IDs are not in the
# "geneid_N" simulation format, so of the original CLUSTERERS list above we
# restrict real-data analysis to only the methods that were explicitly
# requested: CD-HIT, DIAMOND, and MMseqs2. This list is only consumed by the
# new *_realdata functions below; it never touches the simulations code path.
REAL_DATA_CLUSTERERS = ["cdhit", "mmseqs2", "diamond", "panaroo"]

SEQTYPES = ["nt", "aa"]
PARAMORDER = ["st", "c"]
DEFAULT_PARAMS = {"st": "nt", "c": 0.9}
AXIS_TITLE_FONT_SIZE = 10
BASE_FONT_SIZE = 7
DOPREM = True

SKETCH_METHOD_NAMES = ["hdbscan_dist", "hdbscan_tsne", "hdbscan_umap"]

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
    "embed_hdbscan_raw/aa",
    "embed_hdbscan_tsne/aa",
    "embed_hdbscan_umap/aa",
]

SKETCH_FOOTNOTE = (
    "* Sketch/HDBSCAN methods run once per seed on a fixed embedding; "
    "there is no c (minimum sequence identity) sweep, so no averaging over c is performed."
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

    "embed_hdbscan_raw/aa": "#F2A03D",
    "embed_hdbscan_tsne/aa": "#D9770B",
    "embed_hdbscan_umap/aa": "#F7C177",
}


def nicesp(uglysp):
    return " ".join(uglysp.split(".")).capitalize()


def get_font_properties(args):
    return (
        FontProperties(fname=args.font_regular),
        FontProperties(fname=args.font_italic),
        FontProperties(fname=args.font_bold),
    )


def get_param_dict_from_splits(thesplits):
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
    """Returns (genes, labels): for each gene (column) that was actually
    assigned to a cluster by this clusterer, its gene id and the cluster
    label it was assigned to. Genes with no cluster assignment at all
    (e.g. dropped/filtered by the clusterer, left as -1 everywhere) are
    simply skipped rather than forced into a fake cluster."""
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

# counting how many genes belonging to that true class fall into each predicted cluster
# values from 0 to 1 : the higher the better

def get_purity(inlab, truthdf, gene_ids=None):
    # Recode les labels pour qu'ils soient consécutifs
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

    _, ari_p = permutation_test_agreement(
        truthlab_matched,
        probelab_matched,
        metrics.adjusted_rand_score,
        nperm=10000
    )
    
    outlist = [
        True,
        infotuple[0],
        infotuple[1],
        infotuple[2],
        float(metrics.adjusted_rand_score(truthlab_matched, probelab_matched)),
        get_purity(probelab_matched, truthdf_matched, matched_genes),
        float(adjusted_mutual_info_score(truthlab_matched, probelab_matched)),
        ari_p,
    ]
    outlist += [float(el) for el in metrics.homogeneity_completeness_v_measure(truthlab_matched, probelab_matched)]
    return outlist


def permutation_test_agreement(labels1, labels2, metric_function=metrics.adjusted_rand_score, nperm=10000, seed=None):
    rng = default_rng(seed)
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)

    observed = metric_function(labels1, labels2)

    permuted = np.empty(nperm)
    for i in range(nperm):
        permuted[i] = metric_function(rng.permutation(labels1), labels2)

    n_greater = np.sum(permuted >= observed)
    pvalue = float((n_greater + 1) / (nperm + 1))
    return observed, pvalue


def parse_cdhit_identity(line):
    identity = line.strip().split("at", 1)[1].strip().replace("%", "")
    if "/" in identity:
        identity = identity.split("/")[-1]
    return float(identity) / 100.0


def get_df_from_clusterer(clusterer, folderpath, true_max_gene=None):
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


# === NEW (real-data support) ===================================
def get_df_from_clusterer_realdata(clusterer, folderpath):
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

    # clusterer == "panaroo":
    #
    # === CHANGED (bugfix): parse gene_presence_absence.csv instead of the
    # combined_protein_cdhit_out.txt.clstr file =========================
    # The previous version of this branch parsed Panaroo's raw CD-HIT-format
    # clustering file (combined_protein_cdhit_out.txt.clstr) and remapped
    # each internal protein id back to a gene id via gene_data.csv. That was
    # the WRONG input file for real data: it does not reflect Panaroo's
    # final, post-processed gene clusters, and the corrected analysis must
    # instead use Panaroo's actual output table, gene_presence_absence.csv,
    # which is the file Panaroo itself reports as its final pangenome
    # clustering.
    #
    # gene_presence_absence.csv format (one row per gene cluster):
    #   - columns "Gene", "Non-unique Gene name", "Annotation": metadata,
    #     ignored here.
    #   - every other column is one isolate; the cell value is empty when
    #     that cluster is absent from that isolate ("expected, continue
    #     reading the row"), and otherwise contains the gene id(s) (locus
    #     tags) present in that isolate for this cluster. Panaroo separates
    #     multiple paralogous gene ids in the same cell with ";", so we
    #     split on that (and, defensively, on tabs too).
    #
    # The file path is unchanged (still under "panaroo/" inside folderpath).
    #
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
    # explode() preserves (and duplicates) the pre-explode row index, which
    # trips up pd.crosstab's internal reindex; the row index carries no
    # meaning here anyway (cluster_id/gene_id are separate columns), so
    # just reset it.
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
    # Make sure every cluster row is present even if (unexpectedly) it had
    # no genes at all in any isolate, so row count still matches the
    # original table.
    outdf = outdf.reindex(index=gene_presence_absence.index.tolist(), fill_value=-1.0)
    outdf.index.name = "cluster_id"
    return outdf



def get_dfs_from_sketch(folderpath, true_max_gene=None):
    
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


def get_dfs_from_embeddings(folderpath, true_max_gene=None):

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
    
    member_counts = (thedf >= 0.0).sum(axis=1)
    return int((member_counts == 1).sum())

def count_pairs_clusters(thedf):
    
    member_counts = (thedf >= 0.0).sum(axis=1)
    return int((member_counts == 2).sum())

def get_time_diff_from_file(inpath):
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
    with open(inpath, "r") as f:
        for line in f:
            return line.strip()
    return ""


def check_status_of_folder(clusterer, path):
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


def get_info_from_folder(theargs):
    thedir, theass, theseed, datapath = theargs
    truthpath = get_truth_matrix_path(datapath, theass, theseed)
    truthmatrix = pd.read_csv(truthpath, sep="\t")
    truthmatrix = truthmatrix.set_index("gene_id")
    truthlabels = list(truthmatrix["original_gene"])
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


# === NEW (real-data support) ===================================
# === NEW (requirement 2): drop Panaroo "refound" genes before any
# clustering-agreement metric (AMI/ARI/purity/v-measure/F1) is computed ===
def is_refound_gene_id(gene_id):
    """True if `gene_id` looks like a Panaroo-added "refound" gene id
    (Panaroo encodes this directly in the gene id string it writes into
    gene_presence_absence.csv, e.g. "NLEIDEKG_01145_refound_1")."""
    return "refound" in str(gene_id)


def filter_refound_genes(genes, labels):
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
    """Worker for a single clusterer output folder in real-data mode
    (e.g. "mmseqs2_st-aa_c-0.9/"). Pulled out of get_info_from_folder_realdata
    so it can be dispatched to a multiprocessing Pool -- see the
    parallelization note in get_info_from_folder_realdata's docstring.

    Returns None if this folder should be skipped (wrong clusterer, bad
    params, disabled combo, missing expected output file), otherwise a
    dict with everything the caller needs to fold into its accumulators.
    """
    folderpath, folder_name, theass, theseed = args

    splits = folder_name.split("_")
    tmpclusterer = splits[0]
    if tmpclusterer not in REAL_DATA_CLUSTERERS:
        # Requirement 5: restrict real-data analysis to cdhit/mmseqs2/diamond/panaroo.
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
    tmpseqtype = paramdict.get("st", "aa" if tmpclusterer == "panaroo" else DEFAULT_PARAMS["st"])
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
    if not check_status_of_folder(tmpclusterer, folderpath):
        return None

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
    paramlist = [
        paramdict[el] if el in paramdict else (tmpseqtype if el == "st" else DEFAULT_PARAMS[el])
        for el in PARAMORDER
    ]

    # Truth-dependent columns (ARI, purity, AMI, v-measure, ...) do not
    # exist for real data -> NaN, but the row shape is kept identical to
    # calculate_values_from_cluster_matrix's output so
    # build_results_dataframe needs no changes.
    row = (
        [False, theass, theseed, tmpclusterer,
         np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        + [n_clusters, n_singletons, n_pairs]
        + paramlist
        + [runtime]
    )

    genes_i, labels_i = get_labels_list_from_df(thedf)

    # === NEW (requirement 2): strip Panaroo "refound" genes before they
    # can reach any clustering-agreement metric (AMI/ARI/purity/v-measure/
    # F1). Only panaroo's real-data matrix can contain refound gene ids
    # (see get_df_from_clusterer_realdata), so this is a no-op (n_refound
    # == 0) for cdhit/mmseqs2/diamond. ===
    if tmpclusterer == "panaroo":
        genes_i, labels_i, n_refound = filter_refound_genes(genes_i, labels_i)
    else:
        n_refound = 0

    # Only keep the default-c run for the pairwise method-vs-method
    # comparisons, exactly as the simulation path does.
    c_value = paramlist[PARAMORDER.index("c")]
    combo_key = f"{tmpclusterer}/{tmpseqtype}" if c_value == DEFAULT_PARAMS["c"] else None

    return {
        "row": row,
        "genes": genes_i,
        "labels": labels_i,
        "combo_key": combo_key,
        # === NEW (requirement 3B): number of Panaroo-added (refound)
        # genes for this combo, only ever non-zero for panaroo. Only
        # meaningful when combo_key is not None (default-c run), same
        # scoping as genes/labels above.
        "n_refound": n_refound,
    }


def get_info_from_folder_realdata(theargs, nthreads=1):
    """Real-data equivalent of get_info_from_folder.

    === CHANGE: flat layout, no assembly/seed subdirectories ===========
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

    === CHANGE (parallelization) =======================================
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
        adj_mutual_info, adj_rand_index_p, homogeneity, completeness,
        v_measure) are filled with NaN so the returned row still matches
        the column layout build_results_dataframe expects -- this lets us
        reuse that function, and any plot that only reads runtime/
        n_clusters/params, completely unchanged.
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

    # === CHANGE: clusterer param-folders sit directly inside `thedir`
    # (runfolder/real_data), not inside a <assembly>/<seed>/ subpath ===
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
    # === NEW (requirement 3B): combo -> number of Panaroo-added (refound)
    # genes, for the gene-addition boxplot. Only ever populated for the
    # panaroo combo_key; every other combo simply never gets an entry
    # (equivalent to 0 added genes).
    n_refound_out = {}
    all_genes_seen = set()
    for result in raw_results:
        if result is None:
            continue
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


def add_sketch_bracket(ax, x, positions, bar_width=0.5, y_top=-0.16, y_bottom=-0.18,
                        fontprops=None, fontsize=None, row_gap=0.06):
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


def plotter(theargs):
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

    if outnamescaff!= "runtime" :
        # change to log scale
        ax.set_yscale("log")
        #ax.set_ylim(0.97, 1.001)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)

    plt.text(0, 1.01, namedict[assembly], fontproperties=ibmplexsansitalics, horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)
    plt.legend(loc="best", frameon=False, prop=ibmplexsans, handlelength=0.5, handletextpad=0.75, labelspacing=0.3)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(outfolder, "_".join(["plot_c", datatype, assembly, outnamescaff]) + "." + ext),
            bbox_inches="tight",
        )
    fig.clf()
    del fig, ax

# here we fix a c in comparaison to the previous plots
# now we compare clustering methods between them for this fixed c

def plotter_pointplots(theargs):
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

    x_fancy = [FANCYDICT[value] for value in x]
    ymean = []
    ystd = []
    ycount = []
    for x_value in x:
        tmpdf = subdf[
            (subdf.index.get_level_values("simulations") == (datatype == "simulations"))
            & (subdf.index.get_level_values("assembly") == assembly)
            & (subdf.index.get_level_values("clusterer") == x_value.split("/")[0])
            & (subdf["st"] == x_value.split("/")[1])
        ][name].astype(float)
        ymean.append(tmpdf.mean())
        ycount.append(tmpdf.count())
        ystd.append(tmpdf.std() if tmpdf.count() >= 2 else 0.0)

    fig = plt.figure(1, dpi=150, figsize=DEFAULT_FIGSIZE)
    ax = fig.subplots()
    positions = list(range(len(x)))
    bar_width = 0.5

    outnamescaff = name.replace(" ", "").replace("#", "NumberOf")

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

    family_footnote = get_family_footnote(x)
    if family_footnote is not None:
        plt.text(
            0.5, -0.30, family_footnote,
            fontproperties=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1,
            horizontalalignment="center", verticalalignment="top", transform=ax.transAxes,
        )

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(outfolder, "_".join(["plot_point", datatype, assembly, outnamescaff]) + "." + ext),
            bbox_inches="tight",
        )
    fig.clf()
    del fig, ax

def number_of_clusters_violin(theargs):
    name, datadf, namedict, outfolder, assembly, datatype, font_props = theargs
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
            0.5, -0.30, family_footnote,
            fontproperties=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1,
            horizontalalignment="center", verticalalignment="top", transform=ax.transAxes,
        )

    outnamescaff = name.replace(" ", "").replace("#", "NumberOf")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(outfolder, "_".join(["plot_violin", datatype, assembly, outnamescaff]) + "." + ext),
            bbox_inches="tight",
        )
    fig.clf()
    del fig, ax


def number_of_clusters_stacked_bar(theargs):
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
        ax, x, positions, bar_width=bar_width, y_top=-0.22, y_bottom=-0.24,
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
        bbox_to_anchor=(0.5, -0.38),   # below the x-axis labels and the bracket
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
            0.5, -0.50, family_footnote,
            fontproperties=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1,
            horizontalalignment="center", verticalalignment="top", transform=ax.transAxes,
        )

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(outfolder, "_".join(["plot_stackedbar", datatype, assembly, "n_clusters"]) + "." + ext),
            bbox_inches="tight",
        )
    fig.clf()
    del fig, ax


def number_of_clusters_stacked_bar_vs_c(theargs):
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
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(outfolder, "_".join(["plot_stackedbar_c", datatype, assembly, "n_clusters"]) + "." + ext),
            bbox_inches="tight",
        )
    fig.clf()
    del fig, ax


def methods_comparison_heatmap(theargs):
    """Heatmap comparing all clustering methods side by side across the main
    agreement-with-truth metrics (mean over seeds, at the default c). This is
    the 'contingency-style' comparison view: rows = methods (in the requested
    order), columns = metrics, colour = mean score."""
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

    mat = np.full((len(x), len(metric_cols)), np.nan)
    for i, combo in enumerate(x):
        clusterer, seqtype = combo.split("/")
        tmpdf = subdf[
            (subdf.index.get_level_values("clusterer") == clusterer)
            & (subdf["st"] == seqtype)
        ]
        for j, metric_col in enumerate(metric_cols):
            mat[i, j] = tmpdf[metric_col].astype(float).mean()

    row_labels = [
        FANCYDICT[c] + (" *" if c.split("/")[0] in SKETCH_METHOD_NAMES or c.split("/")[0] in EMBED_METHOD_NAMES else "") for c in x
    ]

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

    for i in range(len(x)):
        for j in range(len(metric_cols)):
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

    family_footnote = get_family_footnote(x)
    if family_footnote is not None:
        plt.text(
            0.5, -0.14, family_footnote,
            fontproperties=ibmplexsansitalics, fontsize=BASE_FONT_SIZE - 1,
            horizontalalignment="center", verticalalignment="top", transform=ax.transAxes,
        )

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(outfolder, "_".join(["plot_heatmap_methodcomparison", datatype, assembly]) + "." + ext),
            bbox_inches="tight",
        )
    fig.clf()
    del fig, ax


def build_pairwise_ari_matrix(combo_list, labels_by_seed):
    # Matrix is symmetric (ARI(i,j) == ARI(j,i)), so we only ever compute and
    # store the lower triangle (mat[j, i] with j > i). The upper triangle is
    # left as NaN on purpose; plot_pairwise_ari_heatmap masks NaNs to white so
    # only a lower-triangle heatmap is drawn instead of the redundant full grid.
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


# === NEW (requirement 4, real-data only) ===================================
# build_pairwise_f1_matrix (above) already penalises DELETED genes (it uses
# a fixed reference N and TP = |kept_i & kept_j|, so more deletion -> lower
# recall -> lower F1). It has no way to penalise ADDED genes (Panaroo
# refound genes), because deletion-only F1 assumes "a kept gene is always a
# subset of the original N genes" -- which is no longer true once a method
# can add genes that were never in the original annotation.
#
# STATUS: mode="added_as_fp" is now wired into the pipeline, as an
# ADDITIONAL heatmap (plot_pairwise_f1_heatmap_added_as_fp /
# "plot_heatmap_pairwise_gene_retention_f1_added_as_fp_*.png"), generated
# alongside (not instead of) the existing plot_pairwise_f1_heatmap /
# build_pairwise_f1_matrix output, real-data only. Three modes are still
# implemented here for reference / future use, selected via `mode`:
#
#   mode="deleted_only" (default): identical to build_pairwise_f1_matrix's
#       penalised formula. Included here only so this function is a
#       complete, self-contained drop-in replacement if you want it.
#         TP = |kept_i & kept_j|,  FN = N - TP,  FP = 0
#         F1 = 2*TP / (N + TP)
#
#   mode="added_as_fp" (ENABLED, see above): treat every refound/added
#       gene either method introduced as a false positive. This is the
#       most standard precision/recall reading of "added genes are genes
#       that shouldn't be there": a method that hallucinates extra genes
#       should lose precision, exactly the same way a method that drops
#       genes loses recall.
#         TP = |kept_i & kept_j|
#         FP = |added_i| + |added_j|   (refound genes contributed by either
#              method; an added gene can never be a TP since by
#              definition it has no counterpart in the original N)
#         FN = N - TP
#         Precision = TP / (TP + FP)
#         Recall    = TP / (TP + FN) = TP / N
#         F1 = 2 * Precision * Recall / (Precision + Recall)
#
#   mode="net_kept" (still opt-in only, not wired anywhere): simplest
#       option -- just shrink the effective "kept" count of each method by
#       its own number of added genes before computing the ordinary
#       (deletion-penalised) formula, i.e. treat a method that adds K genes
#       as if it had really only kept (kept - K) genes. This is a rougher
#       approximation than "added_as_fp" (it does not distinguish which
#       specific genes were added), but is simpler to explain and to
#       reproduce by hand.
#         TP_effective = max(0, TP - added_i - added_j)
#         F1 = 2 * TP_effective / (N + TP_effective)
def build_pairwise_f1_matrix_with_additions(
    combo_list, labels_by_seed, total_genes_by_seed=None,
    n_added_by_seed=None, mode="deleted_only",
):
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


# === NEW: exact-cluster-match agreement between pan-genome tools ===========
def build_pairwise_exact_match_matrix(combo_list, labels_by_seed):
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

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(
                outfolder,
                "_".join(["plot_heatmap_pairwise_exact_match", datatype, assembly]) + "." + ext,
            ),
            bbox_inches="tight",
        )
    fig.clf()
    del fig, ax


# === NEW: core genome estimation curve (real data) =========================
def extract_strain_from_geneid(gene_id):
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
    """Given one tool's {gene_id: cluster_label} dict, return a list with
    the number of distinct strains represented in each cluster (one entry
    per cluster, unsorted)."""
    clusters = {}
    for gene, label in gene_label_dict.items():
        clusters.setdefault(label, set()).add(extract_strain_from_geneid(gene))
    return [len(strains) for strains in clusters.values()]


def estimate_total_strains(labels_by_seed):
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

    ax.axhline(total_strains, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("cluster rank", fontsize=AXIS_TITLE_FONT_SIZE, fontproperties=ibmplexsans)
    ax.set_ylabel("number of strains in cluster", fontsize=AXIS_TITLE_FONT_SIZE, fontproperties=ibmplexsans)
    ax.set_ylim(bottom=0, top=total_strains * 1.02)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right", fontsize=BASE_FONT_SIZE, frameon=True)
    ax.set_title(
        f"Core genome estimation — {namedict[assembly]} ({datatype})",
        fontproperties=ibmplexsansbold,
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ibmplexsans)
    fig.tight_layout()

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(
                outfolder,
                "_".join(["plot_core_genome_estimation", datatype, assembly]) + "." + ext,
            ),
        )
    plt.close(fig)


# === NEW (requirement 2): percentage of genes deleted, per method & seed ===
def get_realdata_reference_gene_set(real_data_run_dir):
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


# === NEW (requirement 3B, real-data only): percentage of genes ADDED by
# Panaroo (refound genes), per method & seed ===
def compute_gene_addition_dataframe(addition_by_seed, total_genes_by_seed, assembly):
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


# === NEW (requirement 3): boxplot of deleted-gene percentages per method ===
def plot_gene_deletion_boxplot(theargs):
    """Boxplot of the % of genes deleted by each clusterer/seqtype combo,
    with the distribution taken across random seeds.

    X-axis: clusterer/seqtype combo (e.g. "Panaroo", "PanX (AA)")
    Y-axis: % of genes deleted relative to the original dataset
    Each box: distribution of deleted_pct across all seeds for that combo.
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

    # === CHANGE: exclude methods with 0 deleted genes across all samples,
    # since their box plot would show no meaningful information ===
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
        f"Gene deletion across seeds — {namedict[assembly]} ({datatype})",
        fontproperties=ibmplexsansbold,
    )
    fig.tight_layout()

    outpath = os.path.join(
        outfolder, f"plot_boxplot_gene_deletion_pct_{assembly}_{datatype}.png"
    )
    fig.savefig(outpath)
    plt.close(fig)

    return deletion_df


# === NEW (requirement 3, real-data only): combined deleted% / added% boxplot
def plot_gene_deletion_and_addition_boxplot_realdata(theargs):
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
    """Shared plotting code for the lower-triangle method-vs-method heatmaps
    (pairwise ARI and pairwise gene-retention F1). `mat` is expected to
    already have its upper triangle (j > i) set to NaN."""
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

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            os.path.join(outfolder, "_".join([filename_prefix, datatype, assembly]) + "." + ext),
            bbox_inches="tight",
        )
    fig.clf()
    del fig, ax


def plot_pairwise_ari_heatmap(theargs):
   
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
    plot_pairwise_metric_heatmap(theargs, "ami")


def plot_pairwise_purity_heatmap(theargs):
    plot_pairwise_metric_heatmap(theargs, "purity")


def plot_pairwise_vmeasure_heatmap(theargs):
    plot_pairwise_metric_heatmap(theargs, "v_measure")


def plot_pairwise_f1_heatmap(theargs):
    """Lower-triangle heatmap of the pairwise gene-retention F1 score
    between methods: how much agreement there is between two methods' sets
    of *kept* (non-deleted/non-filtered) genes, now penalised (see
    build_pairwise_f1_matrix) for genes deleted relative to the original
    dataset (requirement 1)."""

    # === CHANGE: theargs now also carries total_genes_by_seed (requirement 1) ===
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


# === NEW (requirement 4, real-data only): pairwise F1 heatmap penalised for
# BOTH deleted genes and Panaroo-added (refound) genes, using
# build_pairwise_f1_matrix_with_additions(mode="added_as_fp"). This is a
# separate, additional plot -- it does NOT replace plot_pairwise_f1_heatmap
# above, which is still generated unchanged (deletion-penalised only, or
# plain Dice for real data as before).
def plot_pairwise_f1_heatmap_added_as_fp(theargs):
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
    seeds = []
    with open(seedsfile, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                seeds.append(int(stripped))
    seeds.sort()
    return seeds


def build_results_dataframe(listoflists):
    outdf = pd.DataFrame(
        listoflists,
        columns=[
            "simulations",
            "assembly",
            "seed",
            "clusterer",
            "adj_rand_index",
            "purity",
            "adj_mutual_info",
            "adj_rand_index_p",
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


# === NEW (real-data support) ===================================
def discover_analysis_tasks_realdata(runfolder, seeds):
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
    parser = argparse.ArgumentParser(description="Analyse gene clustering benchmark runs.")
    parser.add_argument("runfolder", default="./")
    # === NEW (real-data support) ===
    # "simulations" (default) preserves the exact original behaviour end to
    # end. "real_data" switches to the ground-truth-free pipeline: no
    # truth-comparison metrics, no gene-id-format assumptions, and analyses
    # restricted to CD-HIT/DIAMOND/MMseqs2.
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

    # === NEW (real-data support) ===
    # The two modes look for a different top-level directory ("simulations"
    # vs "real_data"), matching the requested assembly-name switch, but the
    # rest of the folder layout underneath is unchanged.
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
        # === NEW (real-data support): no ground-truth directory to check ===
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
    # === NEW (requirements 1 & 2): assembly -> {seed: n_original_genes} ===
    # Total number of genes in the ORIGINAL dataset for each assembly/seed,
    # used as the reference/universe for the gene-deletion penalty in the
    # pairwise F1 heatmap and for the per-method deletion-percentage boxplot.
    total_genes_by_assembly = {}
    # === CHANGE (parallelization): real-data mode has only ONE outer task
    # (see discover_analysis_tasks_realdata), so routing it through main()'s
    # outer Pool.map would waste `-j` on a list of length 1. Instead we call
    # get_info_from_folder_realdata directly and let IT parallelize
    # internally, across clusterer folders, using args.nthreads. The
    # simulations path is untouched: it still has one outer task per
    # assembly/seed and benefits from the outer Pool exactly as before.
    # === NEW (requirement 3B): assembly -> {seed: {combo: n_refound}},
    # real-data only. Feeds the new gene-addition boxplot.
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
        # ================= ORIGINAL SIMULATIONS PIPELINE (unchanged) =================
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

        plottingtasks_violin = [
            ("n_clusters", outdf, namedict, args.outfolder, assembly, datatype, font_props)
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

        # === CHANGE: pairwise F1 task now also carries total_genes_by_seed
        # for this assembly, so the heatmap can penalise deleted genes
        # (requirement 1) ===
        plottingtasks_pairwise_f1 = [
            (
                method_labels_by_assembly.get(assembly, {}),
                total_genes_by_assembly.get(assembly, {}),
                namedict, args.outfolder, assembly, datatype, font_props,
            )
            for assembly in assemblies
        ]

        # === NEW (requirements 2 & 3): gene-deletion percentage boxplot task ===
        plottingtasks_gene_deletion = [
            (
                method_labels_by_assembly.get(assembly, {}),
                total_genes_by_assembly.get(assembly, {}),
                namedict, args.outfolder, assembly, datatype, font_props,
            )
            for assembly in assemblies
        ]
        print("\n> Plotting...")
        # === NEW: collect per-assembly deletion-percentage dataframes so we can
        # write a single combined CSV at the end (requirement 2) ===
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
                "clustering_metrics_with_pvalues.txt"
            ),
            sep="\t"
        )

        # === NEW (requirement 2): write out the per-method/per-seed gene
        # deletion percentages, across all assemblies, as a single CSV ===
        if deletion_dfs:
            combined_deletion_df = pd.concat(deletion_dfs, ignore_index=True)
            combined_deletion_df.to_csv(
                os.path.join(args.outfolder, "gene_deletion_percentages.csv"),
                index=False,
            )

    else:
        # ================= NEW: REAL-DATA PIPELINE =================
        # Requirements 4 & 5: only analyses that do not depend on ground
        # truth are run here, and only for CD-HIT/DIAMOND/MMseqs2/Panaroo
        # (the get_info_from_folder_realdata step above already filtered
        # every other clusterer out, so `outdf` only ever contains these
        # here regardless).
        #
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

        plottingtasks_violin = [
            ("n_clusters", outdf, namedict, args.outfolder, assembly, datatype, font_props)
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

        # === NEW: exact-cluster-match heatmap and core-genome estimation
        # curve tasks (real data only). Both reuse the same
        # method_labels_by_assembly[assembly] = {"run": {combo: {gene:label}}}
        # structure as the other pairwise tasks above. ===
        plottingtasks_exact_match = [
            (method_labels_by_assembly.get(assembly, {}), namedict, args.outfolder, assembly, datatype, font_props)
            for assembly in assemblies
        ]
        plottingtasks_core_genome = [
            (method_labels_by_assembly.get(assembly, {}), namedict, args.outfolder, assembly, datatype, font_props)
            for assembly in assemblies
        ]

        # No total_genes_by_seed passed -> build_pairwise_f1_matrix falls
        # back to the plain, non-truth-penalised Dice/F1 score between
        # methods (see its docstring).
        plottingtasks_pairwise_f1 = [
            (
                method_labels_by_assembly.get(assembly, {}),
                None,
                namedict, args.outfolder, assembly, datatype, font_props,
            )
            for assembly in assemblies
        ]

        # === NEW: per-method gene-deletion AND gene-addition check for
        # real data ========================================================
        # Reuses compute_gene_deletion_dataframe (unchanged, same function
        # as the simulations pipeline) plus the new
        # compute_gene_addition_dataframe (requirement 3B); the only
        # real-data-specific piece is where the reference gene count comes
        # from, since there's no ground truth here -- see
        # get_realdata_reference_gene_set (Panaroo's gene_data.csv).
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

            # === NEW (requirement 4, "added_as_fp" mode, now enabled) ===
            # Separate, additional heatmap alongside the existing
            # plot_pairwise_f1_heatmap output (that one is left exactly as
            # it was -- unpenalised, since no total_genes_by_seed is passed
            # to it above). This one penalises BOTH deleted genes (false
            # negatives against the original gene set) AND Panaroo-added/
            # refound genes (false positives), via
            # build_pairwise_f1_matrix_with_additions(mode="added_as_fp").
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
            pool.map(number_of_clusters_violin, plottingtasks_violin)
            pool.map(number_of_clusters_stacked_bar, plottingtasks_stackedbar)
            pool.map(number_of_clusters_stacked_bar_vs_c, plottingtasks_stackedbar_c)
            pool.map(plot_pairwise_ari_heatmap, plottingtasks_pairwise_ari)
            pool.map(plot_pairwise_ami_heatmap, plottingtasks_pairwise_ari)
            pool.map(plot_pairwise_purity_heatmap, plottingtasks_pairwise_ari)
            pool.map(plot_pairwise_vmeasure_heatmap, plottingtasks_pairwise_ari)
            pool.map(plot_pairwise_exact_match_heatmap, plottingtasks_exact_match)
            pool.map(plot_core_genome_curve_realdata, plottingtasks_core_genome)
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

        # Truth-based metric columns are all-NaN in real-data mode; we still
        # write out the table (runtime/n_clusters/params are real and useful),
        # just under a name that doesn't claim "with_pvalues" (there are no
        # p-values here, since there's no ground truth to permutation-test
        # against).
        outdf.to_csv(
            os.path.join(args.outfolder, "clustering_metrics_real_data.txt"),
            sep="\t"
        )

        # === NEW: write out the per-method gene-deletion percentages
        # (relative to Panaroo's gene_data.csv reference) as a single CSV,
        # same pattern as the simulations pipeline's combined CSV ===
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

        # === NEW (requirement 3B): write out the per-method gene-addition
        # percentages (Panaroo "refound" genes, relative to the same
        # gene_data.csv reference) as a single CSV ===
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
