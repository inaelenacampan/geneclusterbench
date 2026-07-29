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
REAL_DATA_CLUSTERERS = ["cdhit", "mmseqs2", "diamond"]

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
    DIAMOND, and MMseqs2 (requirement 5).

    Why this needs to be a separate function rather than a tweak to the
    original one: the original cdhit/mmseqs2/diamond branches above rely on
    two simulation-only assumptions that silently produce wrong results (or
    crash) on real gene IDs:

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

        # Real gene IDs have no guaranteed numeric structure to sort on;
        # a plain lexical sort gives a deterministic column order and has
        # no bearing on any metric computed downstream.
        listofgenes = sorted(setofgenes)
        for cluster in listofclusters:
            row = [cluster]
            for gene in listofgenes:
                row.append(tmpdict[cluster][gene] if gene in tmpdict[cluster] else -1.0)
            listoflists.append(row)
        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + listofgenes)
        return outdf.set_index("cluster_id")

    # mmseqs2 and diamond share an output format: 2 columns, tab separated,
    # (cluster representative gene id, member gene id).
    filename = "mmseqs2_cluster.tsv" if clusterer == "mmseqs2" else "diamond"
    firstdf = pd.read_csv(
        os.path.join(folderpath, filename),
        names=["cluster_id", "gene_id"],
        sep="\t",
    )
    clusterlist = sorted(firstdf["cluster_id"].unique().tolist())
    genelist = sorted(firstdf["gene_id"].unique().tolist())
    cluster_to_genes = firstdf.groupby("cluster_id")["gene_id"].apply(set)

    listoflists = []
    for cluster_index, cluster_id in enumerate(clusterlist):
        tmpset = cluster_to_genes[cluster_id]
        row = [cluster_index] + [
            1.0 if gene in tmpset else -1.0 for gene in genelist
        ]
        listoflists.append(row)

    outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + genelist)
    return outdf.set_index("cluster_id")


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
def get_info_from_folder_realdata(theargs):
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

    listoflists = []
    method_labels_out = {}
    all_genes_seen = set()
    # === CHANGE: clusterer param-folders sit directly inside `thedir`
    # (runfolder/real_data), not inside a <assembly>/<seed>/ subpath ===
    seed_result_dir = thedir
    for folder_name in os.listdir(seed_result_dir):
        folderpath = os.path.join(seed_result_dir, folder_name)
        if not os.path.isdir(folderpath):
            continue

        splits = folder_name.split("_")
        tmpclusterer = splits[0]
        if tmpclusterer not in REAL_DATA_CLUSTERERS:
            # Requirement 5: restrict real-data analysis to cdhit/mmseqs2/diamond.
            continue

        try:
            paramdict = get_param_dict_from_splits(splits[1:]) if len(splits) > 1 else {}
        except ValueError as exc:
            warnings.warn(
                f"Skipping malformed result folder {folderpath}: {exc}",
                RuntimeWarning, stacklevel=2,
            )
            continue

        invalid_params = [key for key in paramdict if key not in DEFAULT_PARAMS]
        if invalid_params:
            warnings.warn(
                f"Skipping result folder {folderpath}; unsupported parameters {invalid_params}",
                RuntimeWarning, stacklevel=2,
            )
            continue

        tmpseqtype = paramdict.get("st", DEFAULT_PARAMS["st"])
        if tmpclusterer == "diamond" and tmpseqtype == "nt":
            warnings.warn(
                f"Skipping disabled diamond+nt result folder {folderpath}",
                RuntimeWarning, stacklevel=2,
            )
            continue
        if not check_status_of_folder(tmpclusterer, folderpath):
            continue

        # === NEW: per-method progress print (real-data mode) ===
        print(f"\t\t- Reading {tmpclusterer} ({tmpseqtype}) from {folderpath} ...")
        thedf = get_df_from_clusterer_realdata(tmpclusterer, folderpath)
        runtime_path = os.path.join(folderpath, "timebenchmark.txt")
        runtime = get_time_diff_from_file(runtime_path) if os.path.isfile(runtime_path) else np.nan
        n_clusters = len(thedf.index)
        n_singletons = count_singleton_clusters(thedf)
        n_pairs = count_pairs_clusters(thedf)
        # === NEW: confirm what was found for this method once parsed ===
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
        listoflists.append(
            [False, theass, theseed, tmpclusterer,
             np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            + [n_clusters, n_singletons, n_pairs]
            + paramlist
            + [runtime]
        )

        genes_i, labels_i = get_labels_list_from_df(thedf)
        all_genes_seen.update(genes_i)
        # Only keep the default-c run for the pairwise method-vs-method
        # comparisons, exactly as the simulation path does.
        c_value = paramlist[PARAMORDER.index("c")]
        if c_value == DEFAULT_PARAMS["c"]:
            method_labels_out[f"{tmpclusterer}/{tmpseqtype}"] = dict(zip(genes_i, labels_i))

    if not listoflists:
        warnings.warn(
            f"No valid real-data clustering outputs found for {theass}/{theseed}; skipping",
            RuntimeWarning, stacklevel=2,
        )

    n_genes_seen = len(all_genes_seen)
    return (listoflists, theass, nameofass, theseed, method_labels_out, n_genes_seen)


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
                    & (subdf.index.get_level_values("simulations") == True)
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
    plt.text(1, 1.01, "Simulations", fontproperties=ibmplexsans, horizontalalignment="right", verticalalignment="bottom", transform=ax.transAxes)
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
            (subdf.index.get_level_values("simulations") == True)
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
    plt.text(1, 1.01, "Simulations", fontproperties=ibmplexsans, horizontalalignment="right", verticalalignment="bottom", transform=ax.transAxes)

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
            (subdf.index.get_level_values("simulations") == True)
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
    plt.text(1, 1.01, "Simulations", fontproperties=ibmplexsans, horizontalalignment="right", verticalalignment="bottom", transform=ax.transAxes)

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
            (subdf.index.get_level_values("simulations") == True)
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
    plt.text(1, 1.01, "Simulations", fontproperties=ibmplexsans,
             horizontalalignment="right", verticalalignment="bottom", transform=ax.transAxes)

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
                (subdf.index.get_level_values("simulations") == True)
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
    plt.text(1, 1.01, "Simulations", fontproperties=ibmplexsans,
             horizontalalignment="right", verticalalignment="bottom", transform=ax.transAxes)

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
    plt.text(1, 1.03, "Simulations", fontproperties=ibmplexsans,
             horizontalalignment="right", verticalalignment="bottom", transform=ax.transAxes)

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


# === NEW (requirement 2): percentage of genes deleted, per method & seed ===
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
    plt.text(1, 1.03, "Simulations", fontproperties=ibmplexsans,
             horizontalalignment="right", verticalalignment="bottom", transform=ax.transAxes)

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
    info_fn = get_info_from_folder if args.mode == "simulations" else get_info_from_folder_realdata
    if args.nthreads <= 1:
        for task in gettinginfotasks:
            tmpout = info_fn(task)
            listoflists += tmpout[0]
            namedict[tmpout[1]] = tmpout[2]
            method_labels_by_assembly.setdefault(tmpout[1], {})[tmpout[3]] = tmpout[4]
            total_genes_by_assembly.setdefault(tmpout[1], {})[tmpout[3]] = tmpout[5]
    else:
        pool = Pool(args.nthreads)
        for result in pool.map(info_fn, gettinginfotasks):
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
        # truth are run here, and only for CD-HIT/DIAMOND/MMseqs2 (the
        # get_info_from_folder_realdata step above already filtered every
        # other clusterer out, so `outdf` only ever contains these three
        # here regardless).
        #
        # What is intentionally NOT run in this branch, and why:
        #   - plotter / plotter_pointplots for adj_rand_index / purity /
        #     adj_mutual_info / v_measure: these are agreement-with-truth
        #     metrics; every value for real data is NaN (no truth), so the
        #     plots would be empty/meaningless. Only "runtime" is kept.
        #   - methods_comparison_heatmap: its columns are exactly those same
        #     truth-based metrics, so it is skipped entirely.
        #   - plot_pairwise_f1_heatmap / plot_gene_deletion_boxplot with the
        #     gene-deletion penalty: that penalty needs total_genes_by_seed
        #     from a ground-truth gene count, which does not exist for real
        #     data (total_genes_by_assembly here is just "genes seen in the
        #     outputs", not a true universe) -- using it as if it were a
        #     ground truth would silently produce a misleading penalised
        #     score, so we deliberately drop the deletion-boxplot analysis
        #     and only run the *un-penalised* pairwise F1/Dice heatmap
        #     (build_pairwise_f1_matrix's documented fallback behaviour when
        #     total_genes_by_seed is not supplied), which is a pure
        #     method-vs-method agreement score and needs no ground truth.
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

        print("\n> Plotting...")
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
            for task in plottingtasks_pairwise_f1:
                plot_pairwise_f1_heatmap(task)
        else:
            pool = Pool(args.nthreads)
            pool.map(plotter_pointplots, plottingtasks_pointplots)
            pool.map(plotter, plottingtasks)
            pool.map(number_of_clusters_violin, plottingtasks_violin)
            pool.map(number_of_clusters_stacked_bar, plottingtasks_stackedbar)
            pool.map(number_of_clusters_stacked_bar_vs_c, plottingtasks_stackedbar_c)
            pool.map(plot_pairwise_ari_heatmap, plottingtasks_pairwise_ari)
            pool.map(plot_pairwise_f1_heatmap, plottingtasks_pairwise_f1)
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

    print("\n> Done!\n")


if __name__ == "__main__":
    main()
