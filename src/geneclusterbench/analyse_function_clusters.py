#!/usr/bin/env python3
"""
cog_functional_summary.py
==========================

Build per-cluster COG functional-composition summary tables for every
clustering method run by submit_gene_clustering_27_.py / analysed by
analyse_gene_clustering_26_.py on the real (Prokka-annotated ERR*) data.

For every cluster produced by every clustering method, this script reports
what fraction of that cluster's genes belong to each COG functional
category (J, C, E, ... "No COG annotation", etc.), in the format:

    method    cluster_id    gene_ids    number_of_genes    COG_function    percentage

--------------------------------------------------------------------------
STEP 0: get the COG accession -> functional-category-letter mapping file
--------------------------------------------------------------------------
The Prokka .tsv annotation files only give a COG *accession* per gene
(e.g. COG0620) -- not its one-letter functional category. That mapping is
NOT derivable from the Prokka output; it has to come from NCBI's COG
database "definitions" file, which is NOT reachable from this sandbox
(only a short allow-list of package-registry domains is open here), so you
need to fetch it yourself on a machine with internet access (or on the
cluster, if it has egress) and point --cog-def-file at it:

    wget https://ftp.ncbi.nih.gov/pub/COG/COG2024/data/cog-24.def.tab

(An older release such as cog-20.def.tab from the COG2020 directory works
identically -- the script only needs columns 1 and 2.)

That file is tab-separated, no header, with (at least) these columns:
    1. COG ID                (e.g. "COG0620")
    2. functional category   (e.g. "E", or "EH" for COGs with more than
                               one assigned category)
    3. COG name / description
    ... (further columns are ignored)

--------------------------------------------------------------------------
Design decisions (see also the docstrings on each function below)
--------------------------------------------------------------------------
* Genes with an EMPTY COG field in the Prokka tsv (no COG hit at all) are
  put in a synthetic bucket "No COG annotation" -- this is deliberately
  kept separate from the real COG category "S - Function unknown", which
  means "this gene DOES have a COG hit, but that COG's own function is
  unknown". Conflating the two would silently overstate how much of a
  cluster's function is "genuinely uncharacterised" vs. "just never got a
  COG hit from Prokka in the first place".
* A COG accession missing from the def-file you supply (i.e. present in
  Prokka's output but not in NCBI's own table -- can happen across COG
  releases) is put in a separate bucket "COG accession not found in
  mapping file", again so it isn't silently conflated with "no COG at
  all" or "S - Function unknown".
* Some COGs are assigned MORE than one functional-category letter (e.g.
  "EH"). By default (--multi-category all) a gene like that is counted
  once towards EVERY one of its categories, which is the standard COG
  convention -- but it means percentages within a cluster can legitimately
  sum to more than 100%. This is documented in a comment written at the
  top of every output file. If you'd rather each gene only ever count
  once per cluster (percentages always sum to 100%), pass
  --multi-category primary, which uses only the first listed letter.
* Cluster membership is read directly from each clustering method's own
  native output file (not the dense (cluster x gene) matrices that
  analyse_gene_clustering_26_.py builds for computing agreement metrics --
  those are unnecessarily memory-hungry for what we need here, which is
  just "list of gene ids per cluster"). The parsers below mirror the
  real-data branches of get_df_from_clusterer_realdata /
  get_dfs_from_sketch_realdata in analyse_gene_clustering_26_.py, so the
  gene ids/cluster ids match 1:1 with what that script reports.
"""

import argparse
import glob
import json
import os
import re
import sys
import warnings
from collections import defaultdict

# --------------------------------------------------------------------------
# Standard NCBI COG functional-category letters -> description.
# This table is fixed/standardised (unlike the COG-accession -> letter
# mapping, which changes between COG releases), so it's safe to hardcode.
# --------------------------------------------------------------------------
COG_LETTER_TO_DESCRIPTION = {
    "J": "Translation, ribosomal structure and biogenesis",
    "A": "RNA processing and modification",
    "K": "Transcription",
    "L": "Replication, recombination and repair",
    "B": "Chromatin structure and dynamics",
    "D": "Cell cycle control, cell division, chromosome partitioning",
    "Y": "Nuclear structure",
    "V": "Defense mechanisms",
    "T": "Signal transduction mechanisms",
    "M": "Cell wall/membrane/envelope biogenesis",
    "N": "Cell motility",
    "Z": "Cytoskeleton",
    "W": "Extracellular structures",
    "U": "Intracellular trafficking, secretion, and vesicular transport",
    "O": "Posttranslational modification, protein turnover, chaperones",
    "C": "Energy production and conversion",
    "G": "Carbohydrate transport and metabolism",
    "E": "Amino acid transport and metabolism",
    "F": "Nucleotide transport and metabolism",
    "H": "Coenzyme transport and metabolism",
    "I": "Lipid transport and metabolism",
    "P": "Inorganic ion transport and metabolism",
    "Q": "Secondary metabolites biosynthesis, transport and catabolism",
    "R": "General function prediction only",
    "S": "Function unknown",
}

NO_COG_LABEL = "No COG annotation"
UNMAPPED_COG_LABEL = "COG accession not found in mapping file"

# Real-data clusterers supported by submit_gene_clustering_27_.py /
# analyse_gene_clustering_26_.py.
# NOTE: "sketch" is temporarily excluded from the default set below (its
# parsing/summarisation is disabled in process_method_folder() for now --
# see the comment there for how to re-enable it). It's kept in this list
# only so --methods sketch would still resolve if someone re-enables it.
REAL_DATA_CLUSTERERS = [
    "cdhit", "mmseqs2", "diamond", "panaroo",
    "ppanggolin", "panta", "panx", "sketch",
]

# Default value used for --methods (excludes sketch while it's disabled).
DEFAULT_METHODS = [m for m in REAL_DATA_CLUSTERERS if m != "sketch"]

# Matches folder names such as "diamond", "diamond_st-aa", "diamond_st-aa_c-0.9"
FOLDER_NAME_RE = re.compile(
    r"^(?P<clusterer>[a-zA-Z0-9]+)(?:_st-(?P<seqtype>nt|aa))?(?:_c-(?P<c>[0-9.]+))?$"
)


# ==========================================================================
# 1. COG accession -> functional category letter(s)
# ==========================================================================

def load_cog_category_map(cog_def_file):
    """Parse NCBI's cog-*.def.tab into {COG_ID: "letters"}, e.g.
    {"COG0620": "E", "COG0745": "TK"}.

    Only columns 1 (COG ID) and 2 (category letters) are used; anything
    else in the file is ignored. Lines that don't parse cleanly are
    skipped with a warning rather than aborting the whole run.
    """
    cog_to_letters = {}
    n_skipped = 0
    with open(cog_def_file, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                n_skipped += 1
                continue
            cog_id, letters = fields[0].strip(), fields[1].strip()
            if not cog_id or not letters:
                n_skipped += 1
                continue
            cog_to_letters[cog_id] = letters
    if n_skipped:
        warnings.warn(
            f"{cog_def_file}: skipped {n_skipped} malformed line(s) while "
            "parsing the COG ID -> category mapping."
        )
    if not cog_to_letters:
        raise RuntimeError(
            f"No COG ID -> category mappings parsed from {cog_def_file}; "
            "check this is a genuine cog-*.def.tab file."
        )
    return cog_to_letters


def cog_id_to_categories(cog_id, cog_to_letters, multi_category="all"):
    """Return the list of category labels (full descriptions, e.g.
    "E - Amino acid transport and metabolism") for one COG accession.

    multi_category:
        "all"     -- return one label per assigned letter (standard COG
                     behaviour; a gene can land in >1 category).
        "primary" -- return only the first listed letter.
    """
    letters = cog_to_letters.get(cog_id)
    if letters is None:
        return [UNMAPPED_COG_LABEL]

    letters = letters if multi_category == "all" else letters[:1]
    labels = []
    for letter in letters:
        desc = COG_LETTER_TO_DESCRIPTION.get(letter)
        label = f"{letter} - {desc}" if desc else f"{letter} - (unrecognised COG category letter)"
        labels.append(label)
    return labels


# ==========================================================================
# 2. Gene -> COG category label(s), built from the Prokka annotation tsvs
# ==========================================================================

def find_annotation_tsv(sample_dir):
    """Locate the single Prokka annotation .tsv inside one ERRxxxx sample
    directory. Prefers a PROKKA_*.tsv if present, otherwise falls back to
    the only .tsv in the directory (raising if there's more than one, since
    we can't then be sure which is the annotation table)."""
    prokka_matches = glob.glob(os.path.join(sample_dir, "PROKKA_*.tsv"))
    if len(prokka_matches) == 1:
        return prokka_matches[0]
    if len(prokka_matches) > 1:
        raise RuntimeError(
            f"Found multiple PROKKA_*.tsv files in {sample_dir}: {prokka_matches}"
        )

    all_tsv = glob.glob(os.path.join(sample_dir, "*.tsv"))
    if len(all_tsv) == 1:
        return all_tsv[0]
    raise RuntimeError(
        f"Could not uniquely identify the annotation .tsv in {sample_dir} "
        f"(found: {all_tsv})"
    )


def get_real_data_sample_dirs(root_dir):
    """Sorted list of ERRxxxx sample directories under root_dir (same
    convention as submit_gene_clustering_27_.py's function of the same
    name)."""
    if not os.path.isdir(root_dir):
        raise RuntimeError(f"Real-data root directory not found: {root_dir}")
    matches = sorted(
        el for el in os.listdir(root_dir)
        if el.startswith("ERR") and os.path.isdir(os.path.join(root_dir, el))
    )
    if not matches:
        raise RuntimeError(f"No ERR* sample directories found in {root_dir}")
    return [os.path.join(root_dir, el) for el in matches]


def build_gene_to_categories(root_dir, cog_to_letters, multi_category="all"):
    """Scan every ERRxxxx/*.tsv annotation file under root_dir and build
    {gene_id (locus_tag): [category_label, ...]}.

    Genes with an empty COG field get [NO_COG_LABEL].
    Locus tags are assumed unique across the whole real-data run (Prokka
    gives every sample its own random locus-tag prefix, e.g.
    "JLDFOEBO_00001"), matching how the clustering tools themselves treat
    gene ids on this dataset. If a duplicate locus tag does turn up across
    samples, this is flagged loudly rather than silently overwritten,
    since it would indicate a real annotation problem worth investigating.
    """
    gene_to_categories = {}
    duplicate_genes = []

    sample_dirs = get_real_data_sample_dirs(root_dir)
    for sample_dir in sample_dirs:
        tsv_path = find_annotation_tsv(sample_dir)
        with open(tsv_path, "r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            try:
                locus_idx = header.index("locus_tag")
                cog_idx = header.index("COG")
            except ValueError as exc:
                raise RuntimeError(
                    f"{tsv_path}: expected 'locus_tag' and 'COG' columns, "
                    f"got header {header}"
                ) from exc

            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                fields = line.split("\t")
                if len(fields) <= max(locus_idx, cog_idx):
                    continue  # short/malformed row, e.g. no COG field at all
                gene_id = fields[locus_idx].strip()
                cog_field = fields[cog_idx].strip()

                if not cog_field:
                    categories = [NO_COG_LABEL]
                else:
                    categories = cog_id_to_categories(
                        cog_field, cog_to_letters, multi_category
                    )

                if gene_id in gene_to_categories:
                    duplicate_genes.append(gene_id)
                    # Prokka emits one row per FEATURE, not per gene: a
                    # locus_tag typically appears once as a "CDS" row (which
                    # is the row that actually carries the COG field) and
                    # can additionally appear as "gene"/"mRNA" rows for the
                    # same locus_tag (e.g. when the annotation was run with
                    # --addgenes/--addmrna, or --compliant which implies
                    # --addgenes). Those extra rows have an empty COG field.
                    # Since rows are processed in file order, blindly doing
                    # `gene_to_categories[gene_id] = categories` here let a
                    # later, COG-less duplicate row silently overwrite the
                    # real annotation already found on the CDS row -- which
                    # is why COG_function ended up empty for ~every gene.
                    # Only let a duplicate row overwrite an existing entry
                    # if it actually carries COG info (or nothing real has
                    # been seen for this gene_id yet).
                    if (
                        categories == [NO_COG_LABEL]
                        and gene_to_categories[gene_id] != [NO_COG_LABEL]
                    ):
                        continue
                gene_to_categories[gene_id] = categories

    if duplicate_genes:
        warnings.warn(
            f"{len(duplicate_genes)} locus tag(s) appeared in more than one "
            f"ERR* sample's annotation tsv (e.g. {duplicate_genes[:5]}); "
            "the last occurrence encountered was kept. This is unexpected "
            "for Prokka-generated locus tags and worth checking."
        )

    return gene_to_categories


# ==========================================================================
# 3. Cluster membership parsers, one per clustering method
#    (mirrors the real-data branches of analyse_gene_clustering_26_.py's
#    get_df_from_clusterer_realdata / get_dfs_from_sketch_realdata, but
#    returns {cluster_id: [gene_id, ...]} directly instead of a dense
#    (cluster x gene) matrix -- much lighter for real-data gene/cluster
#    counts.)
# ==========================================================================

def parse_cdhit(folderpath):
    clstr_path = os.path.join(folderpath, "cdhit.clstr")
    if not os.path.isfile(clstr_path):
        return {}
    clusters = {}
    cluster_id = None
    with open(clstr_path) as fh:
        for line in fh:
            if line.startswith(">"):
                cluster_id = line.replace(">", "").split(" ")[1].strip()
                clusters[cluster_id] = []
            else:
                gene_id = line.strip().split(">")[1].split("...")[0]
                clusters[cluster_id].append(gene_id)
    return clusters


def parse_mmseqs2_or_diamond(folderpath, clusterer):
    filename = "mmseqs2_cluster.tsv" if clusterer == "mmseqs2" else "diamond"
    path = os.path.join(folderpath, filename)
    if not os.path.isfile(path):
        return {}
    clusters = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            rep, gene_id = line.split("\t")[:2]
            clusters[rep].append(gene_id)
    return clusters


def parse_panaroo(folderpath):
    """Parses gene_presence_absence.csv (Panaroo's final pangenome table),
    same file/format rationale as analyse_gene_clustering_26_.py's
    real-data panaroo branch. One row = one cluster; every isolate column
    cell holds ';'-separated paralog gene ids (locus tags), or is empty if
    that cluster is absent from that isolate."""
    gpa_path = os.path.join(folderpath, "panaroo", "gene_presence_absence.csv")
    if not os.path.isfile(gpa_path):
        return {}
    import csv
    clusters = {}
    with open(gpa_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        meta_cols = {"Gene", "Non-unique Gene name", "Annotation"}
        isolate_idx = [i for i, c in enumerate(header) if c not in meta_cols]
        for row_idx, row in enumerate(reader):
            genes = []
            for i in isolate_idx:
                cell = row[i].strip() if i < len(row) else ""
                if not cell:
                    continue
                for token in re.split(r"[;\t]", cell):
                    token = token.strip()
                    if token:
                        genes.append(token)
            clusters[str(row_idx)] = genes
    return clusters


def parse_panta(folderpath):
    clusters_file = os.path.join(folderpath, "panta", "annotated_clusters.json")
    if not os.path.isfile(clusters_file):
        return {}
    with open(clusters_file) as fh:
        data = json.load(fh)
    clusters = {}
    for group, groupinfo in data.items():
        clusters[group] = [raw.rsplit("-", 1)[-1] for raw in groupinfo["gene_id"]]
    return clusters


def parse_ppanggolin(folderpath):
    """Reads gene_families.tsv, which analyse_gene_clustering_26_.py
    generates on the fly via `ppanggolin write_pangenome --families_tsv`.
    We do NOT re-invoke ppanggolin here to keep this script dependency
    light and side-effect free; if analyse_gene_clustering_26_.py (or you)
    hasn't already produced ppanggolin_outputs/gene_families.tsv for this
    run, run:
        ppanggolin write_pangenome -p <folderpath>/ppanggolin/pangenome.h5 \\
            -o <folderpath>/ppanggolin_outputs/ --families_tsv -f
    first, then re-run this script."""
    families_file = os.path.join(folderpath, "ppanggolin_outputs", "gene_families.tsv")
    if not os.path.isfile(families_file):
        return {}
    clusters = defaultdict(list)
    with open(families_file) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            family_id, gene_id = fields[0], fields[1]
            clusters[family_id].append(gene_id)
    return clusters


def parse_panx(folderpath):
    clusters_file = os.path.join(
        folderpath, "protein_faa", "diamond_matches", "allclusters_final.tsv"
    )
    if not os.path.isfile(clusters_file):
        return {}
    clusters = {}
    n_skipped = 0
    with open(clusters_file) as fh:
        for cluster_idx, raw_line in enumerate(fh):
            line = raw_line.strip()
            if not line:
                continue
            genes, corrupted = [], False
            for field in line.split("\t"):
                if "|" not in field:
                    corrupted = True
                    break
                genes.append(field.rsplit("|", 1)[-1])
            if corrupted:
                n_skipped += 1
                continue
            clusters[str(cluster_idx)] = genes
    if n_skipped:
        warnings.warn(f"panx: skipped {n_skipped} corrupted cluster line(s) in {clusters_file}")
    return clusters


def parse_sketch(folderpath):
    """Sketch produces several sub-methods in one file; returns
    {submethod_name: {cluster_id: [gene_id, ...]}}."""
    tsv_path = os.path.join(folderpath, "distance_clustering", "clusters.tsv")
    if not os.path.isfile(tsv_path):
        return {}
    import csv
    per_submethod = defaultdict(lambda: defaultdict(list))
    with open(tsv_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            per_submethod[row["method"]][row["cluster_id"]].append(row["member"])
    return {k: dict(v) for k, v in per_submethod.items()}


CLUSTER_PARSERS = {
    "cdhit": lambda fp: parse_cdhit(fp),
    "mmseqs2": lambda fp: parse_mmseqs2_or_diamond(fp, "mmseqs2"),
    "diamond": lambda fp: parse_mmseqs2_or_diamond(fp, "diamond"),
    "panaroo": lambda fp: parse_panaroo(fp),
    "panta": lambda fp: parse_panta(fp),
    "ppanggolin": lambda fp: parse_ppanggolin(fp),
    "panx": lambda fp: parse_panx(fp),
}


# ==========================================================================
# 4. Discover method-result folders under the clustering run directory
# ==========================================================================

def discover_method_folders(clustering_run_dir):
    """Yields (method_label, clusterer, folderpath) for every recognised
    result folder directly inside clustering_run_dir (the flat
    runfolder/real_data/<clusterer_paramdir>/ layout produced by
    submit_gene_clustering_27_.py --real-data)."""
    for folder_name in sorted(os.listdir(clustering_run_dir)):
        folderpath = os.path.join(clustering_run_dir, folder_name)
        if not os.path.isdir(folderpath):
            continue
        match = FOLDER_NAME_RE.match(folder_name)
        if not match:
            continue
        clusterer = match.group("clusterer")
        if clusterer not in REAL_DATA_CLUSTERERS:
            continue
        # method_label mirrors the folder name (e.g. "diamond_st-aa_c-0.9"),
        # which already uniquely identifies clusterer + seqtype + identity.
        yield folder_name, clusterer, folderpath


# ==========================================================================
# 5. Summary computation and output
# ==========================================================================

def summarise_cluster(gene_ids, gene_to_categories, multi_category):
    """Return list of (category_label, count, percentage) for one cluster's
    gene list, sorted by descending percentage.

    Percentage is count(genes with this category) / total genes in cluster
    * 100. Genes unknown to gene_to_categories entirely (shouldn't happen
    if the annotation tsvs cover every gene the clusterer emitted, but
    guarded against) also fall into NO_COG_LABEL.
    """
    n_genes = len(gene_ids)
    if n_genes == 0:
        return []

    category_counts = defaultdict(int)
    for gene_id in gene_ids:
        categories = gene_to_categories.get(gene_id, [NO_COG_LABEL])
        for cat in categories:
            category_counts[cat] += 1

    rows = [
        (cat, count, 100.0 * count / n_genes)
        for cat, count in category_counts.items()
    ]
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


def write_summary_tsv(out_path, records, multi_category):
    header_comment = (
        "# COG functional composition per cluster per clustering method.\n"
        "# percentage = 100 * (genes in this cluster with this COG category) / (total genes in cluster).\n"
        + (
            "# multi_category=all: a gene whose COG has more than one assigned "
            "category is counted once in EACH of its categories, so percentages "
            "within a cluster can sum to > 100%.\n"
            if multi_category == "all" else
            "# multi_category=primary: only the first listed COG category letter "
            "is used per gene, so percentages within a cluster sum to exactly 100%.\n"
        )
        + f"# '{NO_COG_LABEL}': gene had no COG field in the Prokka annotation tsv.\n"
        + f"# '{UNMAPPED_COG_LABEL}': gene had a COG accession that was not found "
          "in the supplied --cog-def-file.\n"
    )
    with open(out_path, "w") as fh:
        fh.write(header_comment)
        fh.write("method\tcluster_id\tgene_ids\tnumber_of_genes\tCOG_function\tpercentage\n")
        for rec in records:
            fh.write(
                "{method}\t{cluster_id}\t{gene_ids}\t{n}\t{cog}\t{pct:.1f}%\n".format(
                    method=rec["method"],
                    cluster_id=rec["cluster_id"],
                    gene_ids=",".join(rec["gene_ids"]),
                    n=rec["number_of_genes"],
                    cog=rec["COG_function"],
                    pct=rec["percentage"],
                )
            )


def process_method_folder(method_label, clusterer, folderpath, gene_to_categories, multi_category):
    """Yields output records (dicts) for one clusterer result folder."""
    # --- sketch: temporarily disabled -----------------------------------
    # Skipping "sketch" for now (per request). To re-enable, uncomment the
    # block below and remove the early `return` beneath it.
    #
    # if clusterer == "sketch":
    #     submethod_dict = parse_sketch(folderpath)
    #     for submethod, clusters in submethod_dict.items():
    #         full_label = f"{method_label}:{submethod}"
    #         for cluster_id, gene_ids in clusters.items():
    #             for cat, count, pct in summarise_cluster(gene_ids, gene_to_categories, multi_category):
    #                 yield {
    #                     "method": full_label,
    #                     "cluster_id": cluster_id,
    #                     "gene_ids": gene_ids,
    #                     "number_of_genes": len(gene_ids),
    #                     "COG_function": cat,
    #                     "percentage": pct,
    #                 }
    #     return
    if clusterer == "sketch":
        return

    parser = CLUSTER_PARSERS.get(clusterer)
    if parser is None:
        return
    try:
        clusters = parser(folderpath)
    except Exception as exc:  # noqa: BLE001 - report and skip, don't abort the whole run
        warnings.warn(f"Skipping {folderpath} ({clusterer}): {exc}")
        return
    if not clusters:
        return

    for cluster_id, gene_ids in clusters.items():
        for cat, count, pct in summarise_cluster(gene_ids, gene_to_categories, multi_category):
            yield {
                "method": method_label,
                "cluster_id": cluster_id,
                "gene_ids": gene_ids,
                "number_of_genes": len(gene_ids),
                "COG_function": cat,
                "percentage": pct,
            }


def main():
    parser = argparse.ArgumentParser(
        description="Build per-cluster COG functional-composition summary "
                     "TSVs for every real-data clustering method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--real-datapath",
        default="/nfs/research/jlees/campan/data/clustering_benchmarking/2026_07_24_real_data",
        help="Root directory containing ERRxxxx/ Prokka annotation folders.",
    )
    parser.add_argument(
        "--clustering-run-dir", required=True,
        help="The 'real_data' output directory written by "
             "submit_gene_clustering_27_.py --real-data, e.g. "
             "/hps/nobackup/jlees/campan/tmp/clustering_benchmark_real_data_<timestamp>/real_data",
    )
    parser.add_argument(
        "--cog-def-file", required=True,
        help="Path to NCBI's cog-*.def.tab (COG ID -> functional category "
             "letters). See the module docstring for where to obtain it.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write summary TSVs. Defaults to "
             "<real-datapath>/functional_summaries/",
    )
    parser.add_argument(
        "--methods", default=",".join(DEFAULT_METHODS),
        help="Comma-separated subset of clusterers to summarise. "
             "'sketch' is excluded by default (currently disabled -- see "
             "the comment on process_method_folder()).",
    )
    parser.add_argument(
        "--multi-category", choices=["all", "primary"], default="all",
        help="How to handle COGs assigned more than one functional-category "
             "letter (see module docstring).",
    )
    parser.add_argument(
        "--combined-file-name", default="cog_functional_summary.tsv",
        help="Name of the single combined output TSV (all methods).",
    )
    parser.add_argument(
        "--per-method-files", action="store_true",
        help="Also write one summary TSV per method, in addition to the combined file.",
    )
    args = parser.parse_args()

    wanted_methods = set(args.methods.strip().split(","))
    out_dir = args.out_dir or os.path.join(args.real_datapath, "functional_summaries")
    os.makedirs(out_dir, exist_ok=True)

    print(f"> Loading COG ID -> category mapping from {args.cog_def_file} ...")
    cog_to_letters = load_cog_category_map(args.cog_def_file)
    print(f"  {len(cog_to_letters)} COG accessions loaded.")

    print(f"> Building gene -> COG category map from {args.real_datapath} ...")
    gene_to_categories = build_gene_to_categories(
        args.real_datapath, cog_to_letters, args.multi_category
    )
    print(f"  {len(gene_to_categories)} genes annotated.")

    print(f"> Discovering clustering result folders under {args.clustering_run_dir} ...")
    all_records = []
    per_method_records = defaultdict(list)
    n_folders = 0
    for method_label, clusterer, folderpath in discover_method_folders(args.clustering_run_dir):
        if clusterer not in wanted_methods:
            continue
        n_folders += 1
        print(f"  - {method_label} ({clusterer}) ...")
        records = list(
            process_method_folder(
                method_label, clusterer, folderpath, gene_to_categories, args.multi_category
            )
        )
        if not records:
            print(f"    (no cluster output found/parsed, skipping)")
            continue
        all_records.extend(records)
        per_method_records[method_label].extend(records)

    if n_folders == 0:
        raise RuntimeError(
            f"No recognised clusterer result folders found under {args.clustering_run_dir}"
        )
    if not all_records:
        raise RuntimeError("No cluster records were produced -- check the folder/file paths above.")

    combined_path = os.path.join(out_dir, args.combined_file_name)
    write_summary_tsv(combined_path, all_records, args.multi_category)
    print(f"> Wrote combined summary: {combined_path} ({len(all_records)} rows)")

    if args.per_method_files:
        for method_label, records in per_method_records.items():
            safe_name = re.sub(r"[^A-Za-z0-9_.:-]", "_", method_label)
            method_path = os.path.join(out_dir, f"cog_functional_summary_{safe_name}.tsv")
            write_summary_tsv(method_path, records, args.multi_category)
            print(f"  - {method_path} ({len(records)} rows)")


if __name__ == "__main__":
    main()