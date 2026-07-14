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

            ]
SEQTYPES = ["nt", "aa"]
PARAMORDER = ["st", "c"]
DEFAULT_PARAMS = {"st": "nt", "c": 0.9}
AXIS_TITLE_FONT_SIZE = 10
BASE_FONT_SIZE = 8
DOPREM = True

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
    "cdhit/nt": "#900C3F",
    "cdhit/aa": "#182B55",
    "mmseqs2/nt": "#5F4E94",
    "mmseqs2/aa": "#A291C7",
    "diamond/aa" : "#82CBEC",
    "panaroo/aa": "#D94F21",
    "ppanggolin/aa":"#FEBD2B",
    "panta/aa": "#1B9E77",
    "panx/aa" : "#66A61E"
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
    labels = []
    for column in indf.columns:
        tmp = indf[indf[column] >= 0.0].index
        #if len(tmp) > 1:
            #print(tmp)
        labels.append(tmp[0])
    return labels

# counting how many genes belonging to that true class fall into each predicted cluster
# values from 0 to 1 : the higher the better

def get_purity(inlab, truthdf):
    # Recode les labels pour qu'ils soient consécutifs
    le = LabelEncoder()
    inlab_encoded = le.fit_transform(inlab)
    nclusters = len(le.classes_)

    sumofmaxes = 0

    for column in truthdf.columns:
        countlist = [0] * nclusters
        for gene in truthdf[truthdf[column] == True].index:
            gene_index = int(gene.split("_")[1])
            cluster_label = inlab_encoded[gene_index]
            countlist[cluster_label] += 1
        sumofmaxes += max(countlist)
    return float(sumofmaxes) / float(len(truthdf.index))

# multiple statistics : purity from the above function & metrics from skicit learn
# The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples 
# and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings
# The Mutual Information is a measure of the similarity between two labels of the same data.
# for mathematical expression see documentation

def calculate_values_from_cluster_matrix(infotuple, indf, truthlab, truthdf):
    
    probelab = get_labels_list_from_df(indf)

    _, ari_p = permutation_test_agreement(
        truthlab,
        probelab,
        metrics.adjusted_rand_score,
        nperm=10000
    )
    
    outlist = [
        True,
        infotuple[0],
        infotuple[1],
        infotuple[2],
        float(metrics.adjusted_rand_score(truthlab, probelab)),
        get_purity(probelab, truthdf),
        float(adjusted_mutual_info_score(truthlab, probelab)),
        ari_p,
    ]
    outlist += [float(el) for el in metrics.homogeneity_completeness_v_measure(truthlab, probelab)]
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
        if missing:
            new_cluster_id = max(listofclusters) + 1

            row = [new_cluster_id]

            for gene in listofgenes:
                row.append(
                    1.0 if gene in missing else -1.0
                )

            listoflists.append(row)

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
        if missing:
            new_cluster_id = len(familylist)
            missingset = set(missing)
            row = [new_cluster_id] + [
                1.0 if gene in missingset else -1.0 for gene in genelist
            ]
            listoflists.append(row)

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
        if missing:
            new_cluster_id = len(grouplist)
            missingset = set(missing)
            row = [new_cluster_id] + [
                1.0 if gene in missingset else -1.0 for gene in genelist
            ]
            listoflists.append(row)
        outdf = pd.DataFrame(listoflists, columns=["cluster_id"] + genelist)
        return outdf.set_index("cluster_id")

    raise RuntimeError("Clusterer " + clusterer + " not supported!")

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
    print(f"\t- Getting information from {thedir} execution, {theass} assembly, and {theseed} seed")

    speciesfile = os.path.join(thedir, str(theass), "assembly_species.txt")
    if os.path.isfile(speciesfile):
        nameofass = get_species_name(speciesfile)
    else:
        nameofass = ""

    listoflists = []
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
        tmpseqtype = paramdict.get("st", "aa" if tmpclusterer in ("panaroo", "ppanggolin", "panta") else DEFAULT_PARAMS["st"])
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
        if not check_status_of_folder(tmpclusterer, folderpath):
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
    if not listoflists:
        warnings.warn(
            f"No valid clustering outputs found for {theass}/{theseed}; skipping",
            RuntimeWarning,
            stacklevel=2,
        )
    return (listoflists, theass, nameofass)

# plot idea : how does this metric vary with the clustering threshold, averaged over random seeds, for this assembly

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

    fig = plt.figure(1, dpi=150)
    ax = fig.subplots()
    xs = list(set(list(subdf["c"].astype(float))))
    xs.sort()
    clusterers = list(set(list(subdf.index.get_level_values("clusterer"))))

    for seqtype in SEQTYPES:
        availcl = list(set(list(subdf[subdf["st"] == seqtype].index.get_level_values("clusterer"))))
        ynams = [
            el + "/" + seqtype
            for el in clusterers
            if el in availcl and (el + "/" + seqtype) in FANCYDICT and (el + "/" + seqtype) in CONFIGDICT_COLOURS
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
    if DOPREM:
        plt.text(0.5, 1.01, "Preliminary", fontproperties=ibmplexsansbold, horizontalalignment="center", verticalalignment="bottom", transform=ax.transAxes)
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
    x = []
    for clusterer, seqtype in [(c, st) for st in SEQTYPES for c in clusterers]:
        combo = clusterer + "/" + seqtype
        if combo not in FANCYDICT or combo not in CONFIGDICT_COLOURS:
            continue
        if len(list(subdf[(subdf.index.get_level_values("clusterer") == clusterer) & (subdf["st"] == seqtype)][name])):
            x.append(combo)

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

    fig = plt.figure(1, dpi=150)
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
    ax.set_xticklabels(x_fancy)

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
    if DOPREM:
        plt.text(0.5, 1.01, "Preliminary", fontproperties=ibmplexsansbold, horizontalalignment="center", verticalalignment="bottom", transform=ax.transAxes)

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
    x = []
    for clusterer, seqtype in [(c, st) for st in SEQTYPES for c in clusterers]:
        combo = clusterer + "/" + seqtype
        if combo not in FANCYDICT or combo not in CONFIGDICT_COLOURS:
            continue
        if len(list(subdf[(subdf.index.get_level_values("clusterer") == clusterer) & (subdf["st"] == seqtype)][name])):
            x.append(combo)

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

    fig = plt.figure(1, dpi=150)
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
    ax.set_xticklabels(x_fancy)
    ax.set_ylim(1500, 2500)

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
    if DOPREM:
        plt.text(0.5, 1.01, "Preliminary", fontproperties=ibmplexsansbold, horizontalalignment="center", verticalalignment="bottom", transform=ax.transAxes)

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

    clusterers = list(set(list(subdf.index.get_level_values("clusterer"))))
    x = []
    for clusterer, seqtype in [(c, st) for st in SEQTYPES for c in clusterers]:
        combo = clusterer + "/" + seqtype
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

    fig = plt.figure(1, dpi=150, figsize=(max(5, len(x) * 1.4), 5))
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
    ax.set_xticklabels(x_fancy)
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
    if DOPREM:
        plt.text(0.5, 1.01, "Preliminary", fontproperties=ibmplexsansbold,
                 horizontalalignment="center", verticalalignment="bottom", transform=ax.transAxes)

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
        bbox_to_anchor=(0.5, -0.18),   # below the x-axis
        frameon=False,
        prop=ibmplexsans,
        handlelength=0.8,
        handletextpad=0.75,
        labelspacing=0.3,
        ncol=len(method_handles),       # all in one row
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
    clusterers = list(set(list(subdf.index.get_level_values("clusterer"))))
    x = []
    for clusterer, seqtype in [(c, st) for st in SEQTYPES for c in clusterers]:
        combo = clusterer + "/" + seqtype
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

    fig = plt.figure(1, dpi=150, figsize=(max(8, n_c * n_methods * 0.55), 5))
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
    if DOPREM:
        plt.text(0.5, 1.01, "Preliminary", fontproperties=ibmplexsansbold,
                 horizontalalignment="center", verticalalignment="bottom", transform=ax.transAxes)

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


def main():
    parser = argparse.ArgumentParser(description="Analyse gene clustering benchmark runs.")
    parser.add_argument("runfolder", default="./")
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
    if "simulations" not in lsdirs:
        raise RuntimeError("No valid folders found!")
    print("> Getting seeds")
    seeds = load_seeds(seedsfile)
    print("> Got {} seeds".format(len(seeds)))

    print("\n> Getting info...")
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

    listoflists = []
    namedict = {}
    if args.nthreads <= 1:
        for task in gettinginfotasks:
            tmpout = get_info_from_folder(task)
            listoflists += tmpout[0]
            namedict[tmpout[1]] = tmpout[2]
    else:
        pool = Pool(args.nthreads)
        for result in pool.map(get_info_from_folder, gettinginfotasks):
            listoflists += result[0]
            namedict[result[1]] = result[2]
        pool.close()
        pool.join()
    print("\n> Done!")

    if not listoflists:
        raise RuntimeError("No valid clustering results were produced from the analysable simulations")

    outdf = build_results_dataframe(listoflists)
    print("Clusterers found:", set(outdf.index.get_level_values("clusterer")))
    
    assemblies = set(list(outdf.index.get_level_values("assembly")))

    if not os.path.isdir(args.outfolder):
        os.makedirs(args.outfolder)

    plotstodo = ["adj_rand_index", "purity", "adj_mutual_info", "v_measure", "runtime"]
    print("\n> Preparing plotting tasks...")
    plottingtasks_pointplots = [
        (plot_name, outdf, namedict, args.outfolder, assembly, "simulations", font_props)
        for plot_name in plotstodo
        for assembly in assemblies
    ]
    plottingtasks = [
        (plot_name, outdf, namedict, args.outfolder, assembly, "simulations", font_props)
        for plot_name in plotstodo
        for assembly in assemblies
    ]

    plottingtasks_violin = [
        ("n_clusters", outdf, namedict, args.outfolder, assembly, "simulations", font_props)
        for assembly in assemblies
    ]

    plottingtasks_stackedbar_c = [
        ("n_clusters", outdf, namedict, args.outfolder, assembly, "simulations", font_props)
        for assembly in assemblies
    ]

    plottingtasks_stackedbar = [
        ("n_clusters", outdf, namedict, args.outfolder, assembly, "simulations", font_props)
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

    else:
        pool = Pool(args.nthreads)
        pool.map(plotter_pointplots, plottingtasks_pointplots)
        pool.map(plotter, plottingtasks)
        pool.map(number_of_clusters_violin, plottingtasks_violin)
        pool.map(number_of_clusters_stacked_bar, plottingtasks_stackedbar)
        pool.map(number_of_clusters_stacked_bar_vs_c, plottingtasks_stackedbar_c)
        pool.close()
        pool.join()
    
    outdf.to_csv(
        os.path.join(
            args.outfolder,
            "clustering_metrics_with_pvalues.txt"
        ),
        sep="\t"
    )

    print("\n> Done!\n")


if __name__ == "__main__":
    main()
