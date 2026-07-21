# import libraries
from Bio import SeqIO
import argparse
import os
import time
from pathlib import Path
import glob

# new data paths
DEFAULT_DATAPATH = (
    "/nfs/research/jlees/campan/data/clustering_benchmarking/"
    "2026_06_10_simsnowwithntandaasandgffs"
)
# updated
DEFAULT_SOFTWAREDIR = (
    "/hps/software/users/jlees/campan/clustering_benchmarking/software"
)
# run-benchmark to be written by myself
DEFAULT_RUNNER = (
    "/hps/software/users/jlees/campan/assembler_development/"
    "benchmarking/run_benchmark.py"
)
# path to the geneclusterbench repo checkout (holds pyproject.toml/uv.lock),
# used to invoke cluster_distance_file.py via `uv run --project`
DEFAULT_GCB_REPO = "/hps/software/users/jlees/campan/clustering_benchmarking"
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_SEEDS = str(PROJECT_ROOT / "data" / "random_numbers.txt")

MMSEQS2_SCAFFOLD = (
    "mkdir -p {workdir} && mkdir -p {tmpdir} && cd {workdir} && "
    "inittime=$(date +'%d/%m/%Y-%H:%M:%S') && "
    "{execexec} easy-cluster {inputfile} {outputfile} {tmpdir} "
    "--min-seq-id {c} --threads {ncores} && "
    "echo $inittime'=>'$(date +'%d/%m/%Y-%H:%M:%S') > timebenchmark.txt && cd -"
)

CDHIT_SCAFFOLD = (
    "mkdir -p {workdir} && cd {workdir} && "
    "inittime=$(date +'%d/%m/%Y-%H:%M:%S') && "
    "{execexec} -i {inputfile} -M {mem}M -n {word_size} -c {c} -d 0 -T {ncores} "
    "-o {outputfile} && "
    "echo $inittime'=>'$(date +'%d/%m/%Y-%H:%M:%S') > timebenchmark.txt && cd -"
)

DIAMOND_SCAFFOLD = (
    "mkdir -p {workdir} && cd {workdir} && "
    "inittime=$(date +'%d/%m/%Y-%H:%M:%S') && "
    "{execexec} cluster -d {inputfile} -o {outputfile} "
    "--approx-id {approx_id} --threads {ncores} -M {mem}G && "
    "echo $inittime'=>'$(date +'%d/%m/%Y-%H:%M:%S') > timebenchmark.txt && cd -"
)

PANAROO_SCAFFOLD = (
    "mkdir -p {workdir} && cd {workdir} && "
    "inittime=$(date +'%d/%m/%Y-%H:%M:%S') && "
    "{execexec} -i {inputfile} -o {outdir} "
    "-c {c} --clean-mode strict -t {ncores} && "
    "echo $inittime'=>'$(date +'%d/%m/%Y-%H:%M:%S') > timebenchmark.txt && cd -"
)

PANTA_SCAFFOLD = (
    "mkdir -p {workdir} && cd {workdir} && "
    "inittime=$(date +'%d/%m/%Y-%H:%M:%S') && "
    "{execexec} main -g {inputfile} -o {outdir} "
    "-i {c} -t {ncores}&& "
    "echo $inittime'=>'$(date +'%d/%m/%Y-%H:%M:%S') > timebenchmark.txt && cd -"
)

PPANGGOLIN_SCAFFOLD = (
    "mkdir -p {workdir} && cd {workdir} && "
    "{execexec} annotate --anno {inputfile} -o {outdir} --cpu {ncores} -f && "
    "inittime=$(date +'%d/%m/%Y-%H:%M:%S') && "
    "{execexec} cluster -p {outdir}/pangenome.h5 --identity {c} --cpu {ncores} && "
    "echo $inittime'=>'$(date +'%d/%m/%Y-%H:%M:%S') > timebenchmark.txt && cd -"
)

PANX_SCAFFOLD = (
    "mkdir -p {workdir} && mkdir -p {gbkdir} && cd {workdir} && "
    "inittime=$(date +'%d/%m/%Y-%H:%M:%S') && "
    "for f in {inputfiles}; do ln -sf $f {gbkdir}/$(basename $f); done && "
    "export PATH={envbindir}:$PATH && "
    "{execexec} -fn {workdir} -sl {species} -st 1 3 4 5 6 -t {ncores} -dmi {dmi} && "
    "echo $inittime'=>'$(date +'%d/%m/%Y-%H:%M:%S') > timebenchmark.txt && cd -"
)

# removed concat fasta? good / bad idea?
SKETCH_SCAFFOLD_AA = (
    "mkdir -p {workdir} && mkdir -p {sketchdir} && "
    "cd {workdir} && "
    "inittime=$(date +'%d/%m/%Y-%H:%M:%S') && "
    "{execexec} sketch -f {inputfile} -o {outputprefix} "
    "-s 64 -k 5 --seq-type aa --threads {ncores} -v && "
    "{execexec} dist {outputprefix} -o {distoutput} -k 5 --threads {ncores} -v && "
    "uv run --project {gcbrepo} python -m geneclusterbench.cluster_distance_file "
    "--dist-file {distoutput} --nthreads {ncores} && "
    "echo $inittime'=>'$(date +'%d/%m/%Y-%H:%M:%S') > timebenchmark.txt && cd -"
)

SKETCH_SCAFFOLD_NT = (
    "mkdir -p {workdir} && mkdir -p {sketchdir} && "
    "cd {workdir} && "
    "inittime=$(date +'%d/%m/%Y-%H:%M:%S') && "
    "{execexec} sketch -f {inputfile} -o {outputprefix} "
    "-s 64 -k 17 --seq-type dna --threads {ncores} -v && "
    "{execexec} dist {outputprefix} -o {distoutput} -k 17 --threads {ncores} -v && "
    "uv run --project {gcbrepo} python -m geneclusterbench.cluster_distance_file "
    "--dist-file {distoutput} --nthreads {ncores} && "
    "echo $inittime'=>'$(date +'%d/%m/%Y-%H:%M:%S') > timebenchmark.txt && cd -"
)

# ProstT5 embeddings + HDBSCAN/UMAP/t-SNE clustering. Only sensible for AA
# input (ProstT5 expects amino-acid, or lower-case 3Di, sequences).
EMBEDDINGS_SCAFFOLD = (
    "mkdir -p {workdir} && mkdir -p {embeddir} && cd {workdir} && "
    "inittime=$(date +'%d/%m/%Y-%H:%M:%S') && "
    "uv run --project {gcbrepo} python -m geneclusterbench.embedprots "
    "--input-fasta {inputfile} --out-pk {pkoutput} --nthreads {ncores} && "
    "uv run --project {gcbrepo} python -m geneclusterbench.cluster_embeddings_file "
    "--embeddings-file {pkoutput} --nthreads {ncores} --out-dir {clusterdir} && "
    "echo $inittime'=>'$(date +'%d/%m/%Y-%H:%M:%S') > timebenchmark.txt && cd -"
)

SLURM_SCAFFOLD = (
    "sbatch --array={arrayvals} -c {nth} -t {timemax} --mem {memmax}G "
    "-J {jobname} -e {logpath}/log.%A.%a.%x.err "
    "-o {logpath}/log.%A.%a.%x.out --wrap '{command}' {other}"
)
EXEC_SCAFFOLD = "python3 {executable} {filepath}"

CRANGE = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
DEFAULT_PARAMS = {"c": 0.9}
COMMANDS_FILE = "execcommands.tsv"
CDHIT_EST_MIN_C = 0.8

# basic reading function for the random seeds
def load_seeds(seedsfile):
    seeds = []
    with open(seedsfile, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                seeds.append(int(stripped))
    seeds.sort()
    return seeds

# type of "alphabet" used : aminoacids or nucleotide
# thresholds : diffrent values how were there chosen?

"""

The core idea: CD-HIT uses a short-word filtering heuristic to quickly decide whether two sequences are likely 
to meet the identity threshold before doing a full alignment. It looks for shared k-mers (words) of length n. 
The math is: if two sequences share ≥ c (identity), they must share at least some minimum number of common k-mers of size n. 
The smaller the word size, the more permissive/sensitive the filter, but also the slower and more memory-intensive 
(more possible words to index).

"""
def get_cdhit_word_size(c, seqtype):
    if seqtype == "aa":
        if c >= 0.7:
            return 5
        if c >= 0.6:
            return 4
        if c >= 0.5:
            return 3
        return 2
    if seqtype == "nt":
        if c >= 0.9:
            return 8
        if c >= 0.88:
            return 7
        if c >= 0.85:
            return 6
        if c >= 0.8:
            return 5
        return 4
    raise RuntimeError("Not supported sequence type " + seqtype)


def get_c_values_for_process(proc, seqtype):
    if proc == "sketch":
        # sketchlib has no identity threshold; run once per seed with the
        # default "c" value so no _c-suffix is added to the output folder.
        return [DEFAULT_PARAMS["c"]]
    if proc == "embeddings":
        # ProstT5 embeddings only make sense for AA input, and (like sketch)
        # there is no identity threshold, so run once per seed with the
        # default "c" value and skip nt entirely.
        if seqtype != "aa":
            return []
        return [DEFAULT_PARAMS["c"]]
    if proc == "cdhit" and seqtype == "nt":
        return [c for c in CRANGE if c >= CDHIT_EST_MIN_C]
    if proc == "diamond" and seqtype == "nt":
        return []
    if proc == "panaroo" and seqtype == "nt":
        return []
    if proc == "panta" and seqtype == "nt":
        return []
    return CRANGE

def get_or_write_ppanggolin_anno_list(simdir):
    listpath = os.path.join(simdir, "ppanggolin_anno_list.tsv")
    if not os.path.isfile(listpath):
        gffs = sorted(el for el in os.listdir(simdir) if el.endswith(".gff"))
        if not gffs:
            raise RuntimeError(f"No GFF files found in {simdir}")
        with open(listpath, "w") as handle:
            for gff in gffs:
                genome_name = os.path.splitext(gff)[0]
                handle.write(f"{genome_name}\t{os.path.join(simdir, gff)}\n")
    return listpath

def write_scketch_list(simdir):
    listpath = os.path.join(simdir, "scketch_nt_list.tsv")
    if not os.path.isfile(listpath):
        gffs = sorted(glob.glob(os.path.join(simdir, "*iso_*.fasta")))
        if not gffs:
            raise RuntimeError(f"No fasta files found in {simdir}")
        with open(listpath, "w") as handle:
            for gff in gffs:
                genome_name = os.path.splitext(gff)[0]
                handle.write(f"{genome_name}\t{os.path.join(simdir, gff)}\n")
    return listpath

def get_gene_list_for_sketch(simdir, seqtype):
    fasta_path = get_clustering_fasta(simdir, seqtype)

    genes_dir = os.path.join(simdir, f"sketch_genes_{seqtype}")
    os.makedirs(genes_dir, exist_ok=True)

    tsv_path = os.path.join(simdir, f"_sketch_gene_list_{seqtype}.tsv")
    if not os.path.isfile(tsv_path):
        with open(tsv_path, "w") as tsv:
            for record in SeqIO.parse(fasta_path, "fasta"):
                gene_fasta = os.path.join(genes_dir, record.id + ".fasta")
                if not os.path.isfile(gene_fasta):
                    SeqIO.write([record], gene_fasta, "fasta")
                tsv.write(f"{record.id}\t{gene_fasta}\n")
    return tsv_path

def get_command_for_process(proc, seqtype, infile, outfolder, nthreads, maxmem, softwaredir, c=0.9, gcbrepo=None):
    
    if seqtype == "nt" :
        seq_arg = "nucleotide"
    else :
        seq_arg = "protein"

    # cd-hit method
    if proc == "cdhit":

        cdhitexec = os.path.join(
                softwaredir,
                "cdhit/cdhit/cd-hit-est" if seqtype == "nt" else "cdhit/cdhit/cd-hit",
            )
        
        word_size = get_cdhit_word_size(c, seqtype)
        return CDHIT_SCAFFOLD.format(
            workdir=outfolder,
            inputfile=infile,
            execexec=cdhitexec,
            mem=int(maxmem) * 1000,
            c=c,
            word_size=word_size,
            ncores=nthreads,
            outputfile="./cdhit",
        )
    #mmseqs2 method
    if proc == "mmseqs2":

        mmseqs2exec = os.path.join(softwaredir, "mmseqs/bin/mmseqs")

        return MMSEQS2_SCAFFOLD.format(
            workdir=outfolder,
            inputfile=infile,
            execexec=mmseqs2exec,
            tmpdir=os.path.join(outfolder, "tmp"),
            c=c,
            ncores=nthreads,
            outputfile="./mmseqs2",
        )
    # diamond method
    if proc == "diamond":

        diamondexec = os.path.join(softwaredir, "Diamond/diamond")

        return DIAMOND_SCAFFOLD.format(
            workdir=outfolder,
            inputfile=infile,
            execexec=diamondexec,
            approx_id=int(c * 100), # transform 30% in 30 (as in documentation)
            ncores=nthreads,
            mem=int(maxmem),
            outputfile="./diamond",
        )
    # panaroo method
    if proc == "panaroo":
        
        panarooexec = os.path.join(softwaredir, "panaroo_env/bin/panaroo")

        return PANAROO_SCAFFOLD.format(
            workdir=outfolder,
            inputfile=infile,
            execexec=panarooexec,
            c=c, 
            ncores=nthreads,
            outdir="./panaroo",
        )
    # panta method
    if proc == "panta":
        
        pantaexec = os.path.join(softwaredir, "panta/bin/panta")

        return PANTA_SCAFFOLD.format(
            workdir = outfolder,
            inputfile = infile,
            execexec = pantaexec,
            c = c,
            ncores = nthreads,
            outdir = "./panta",
        )
    
    # ppanggolin method
    if proc == "ppanggolin":

        ppanggolinexec = os.path.join(softwaredir, "ppanggolin/bin/ppanggolin")

        return PPANGGOLIN_SCAFFOLD.format(
            workdir=outfolder,
            inputfile=infile,
            execexec=ppanggolinexec,
            outdir="./ppanggolin",
            c=c,
            ncores=nthreads,
        )
    
    # panX method
    if proc == "panx":
        # panX requires Python 2.7 (#!/usr/bin/env python2 in the script)
        # plus a dedicated legacy conda env (biopython, diamond, ete2,
        # fasttree, mafft, mcl, treetime==0.6.*, etc. — see
        # panX-environment.yml in the panX repo). The sbatch job's shell
        # won't have any conda env activated, so the shebang's `python2`
        # lookup fails. Call that env's python interpreter explicitly
        # instead of executing panX.py directly.
        panx_python = os.path.join(softwaredir, "panX/bin/python2")
        panx_script = os.path.join(softwaredir, "panX/panX.py")
        panxexec = f"{panx_python} {panx_script}"
        panx_envbin = os.path.join(softwaredir, "panX/bin")

        # panX expects to be pointed (-fn) directly at a folder that
        # contains input_GenBank/ (see panX docs: ./data/YourSpecies/input_GenBank).
        # It does NOT walk into a nested data/<species>/ subfolder, so
        # input_GenBank must live right under outfolder, not under
        # outfolder/data/<species>/.
        species = "simulated_species"
        gbkdir = os.path.join(outfolder, "input_GenBank")

        return PANX_SCAFFOLD.format(
            workdir=outfolder,
            gbkdir=gbkdir,
            inputfiles=infile,
            execexec=panxexec,
            species=species,
            ncores=nthreads,
            envbindir=panx_envbin,
            # -dmi/--diamond_identity is on a 0-100 percentage scale, default
            # 0 (no restriction), same convention as the diamond clusterer's
            # --approx-id above.
            dmi=int(c * 100),
        )
    # sketchlib method (sketch + dist)
    if proc == "sketch":

        sketchexec = os.path.join(softwaredir, "sketchlib.rust/target/release/sketchlib")
        sketchdir = os.path.join(outfolder, "sketch")

        if seqtype == "nt" :
            return SKETCH_SCAFFOLD_NT.format(
                workdir=outfolder,
                sketchdir=sketchdir,
                #analysisdir=analysisdir,
                inputfile=infile,
                execexec=sketchexec,
                outputprefix=os.path.join(sketchdir, "sketch"),
                distoutput=os.path.join(sketchdir, "output.dist"),
                gcbrepo=gcbrepo,
                ncores=nthreads,
            )
        else :
            return SKETCH_SCAFFOLD_AA.format(
                workdir=outfolder,
                sketchdir=sketchdir,
                #analysisdir=analysisdir,
                inputfile=infile,
                execexec=sketchexec,
                outputprefix=os.path.join(sketchdir, "sketch"),
                distoutput=os.path.join(sketchdir, "output.dist"),
                gcbrepo=gcbrepo,
                ncores=nthreads,
            )
        
    # ProstT5 embeddings method (embedprots + cluster_embeddings_file)
    if proc == "embeddings":

        embeddir = os.path.join(outfolder, "embeddings")

        return EMBEDDINGS_SCAFFOLD.format(
            workdir=outfolder,
            embeddir=embeddir,
            inputfile=infile,
            pkoutput=os.path.join(embeddir, "embeddings.pk"),
            clusterdir=os.path.join(outfolder, "clustering"),
            gcbrepo=gcbrepo,
            ncores=nthreads,
        )

    raise RuntimeError("Process " + proc + " not supported")


def get_clustering_fasta(simdir, seqtype):
    if seqtype == "nt":
        expected = "*_for_clustering.fasta"
        matches = [
            el for el in os.listdir(simdir)
            if el.endswith("_for_clustering.fasta")
        ]
    elif seqtype == "aa":
        expected = "*_for_clustering_aa.fasta"
        matches = [
            el for el in os.listdir(simdir)
            if el.endswith("_for_clustering_aa.fasta")
        ]
    else:
        raise RuntimeError("Not supported sequence type " + seqtype)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {expected} in {simdir}, found {len(matches)}"
        )
    return os.path.join(simdir, matches[0])

def get_fasta_file_list(simdir):
    matches = [
        el for el in os.listdir(simdir)
        if el.endswith("_fasta_file.tsv")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one *_fasta_file.tsv in {simdir}, found {len(matches)}"
        )
    return os.path.join(simdir, matches[0])

def get_sim_iso_gffs(simdir):
    matches = [
        os.path.join(simdir, el) for el in os.listdir(simdir)
        if el.endswith(".gff")
    ]
    if not matches:
        raise RuntimeError(f"No GFF files found in {simdir}")
    return " ".join(matches)  # e.g. "path/iso_0.gff path/iso_1.gff ..."


def fix_genbank_metadata(gbk_file):
    records = []

    for record in SeqIO.parse(gbk_file, "genbank"):
        source_found = False

        for feature in record.features:
            if feature.type == "source":
                source_found = True
                if "organism" not in feature.qualifiers:
                    feature.qualifiers["organism"] = ["unknown"]
                if "strain" not in feature.qualifiers:
                    feature.qualifiers["strain"] = ["unknown"]

            if feature.type == "CDS":
                if "product" not in feature.qualifiers:
                    feature.qualifiers["product"] = ["hypothetical_protein"]
                if "translation" not in feature.qualifiers:
                    feature.qualifiers["translation"] = [
                        str(feature.extract(record.seq).translate())
                    ]

        if not source_found:
            source = SeqFeature(
                FeatureLocation(0, len(record)),
                type="source",
                qualifiers={"organism": ["unknown"], "strain": ["unknown"]}
            )
            record.features.insert(0, source)

        records.append(record)

    tmp_file = gbk_file + ".tmp"
    SeqIO.write(records, tmp_file, "genbank")
    os.replace(tmp_file, gbk_file)  # atomic on POSIX filesystems

def get_sim_iso_gbks(simdir):
    matches = [
        os.path.join(simdir, el) for el in os.listdir(simdir)
        if el.endswith(".gbk") or el.endswith(".gb")
    ]
    if not matches:
        raise RuntimeError(
            f"No GenBank (.gbk/.gb) files found in {simdir}; "
            "panX requires annotated GenBank input, not GFF."
        )
    for gbk in matches:
        fix_genbank_metadata(gbk)
    return " ".join(matches)  # e.g. "path/iso_0.gbk path/iso_1.gbk ..."

def submit_clustering_jobs(args):
    print("> Getting seeds")
    # reads a list of random seeds : each seed corresponds to one simulated dataset
    seeds = load_seeds(args.seeds)
    print("> Got {} seeds".format(len(seeds)))

    print("\n> Preparing jobs...")
    jobinfo = []
    # reproducibility
    timestamp = int(time.time()) if args.preset_timestamp < 0 else args.preset_timestamp
    generaloutdir = os.path.join(args.temp_outdir, f"clustering_benchmark_{timestamp}")

    simulations_dir = os.path.join(args.datapath, "simulations")
    assemblies = [
        el for el in os.listdir(simulations_dir)
        if os.path.isdir(os.path.join(simulations_dir, el))
    ]

    for assembly in assemblies:
        for seed in seeds:
            simdir = os.path.join(simulations_dir, assembly, str(seed))
            if not os.path.isdir(simdir):
                continue
            for process in args.process:
                
                if process in ("panaroo", "ppanggolin", "panta", "panx"):
                    if process == "ppanggolin":
                        infile = get_or_write_ppanggolin_anno_list(simdir)
                    elif process == "panx":
                        infile = get_sim_iso_gbks(simdir)
                    else:
                        infile = get_sim_iso_gffs(simdir)

                    for c_value in get_c_values_for_process(process, "aa"):
                        suffix = f"_c-{c_value}" if c_value != DEFAULT_PARAMS["c"] else ""
                        jobinfo.append(
                            get_command_for_process(
                                process,
                                "aa",
                                infile,
                                os.path.join(
                                    generaloutdir,
                                    "simulations",
                                    assembly,
                                    str(seed),
                                    process + suffix,
                                ),
                                args.threads,
                                args.mem,
                                args.softwaredir,
                                c_value,
                                gcbrepo=args.gcb_repo,
                            )
                        )
                else:
                    for seqtype in args.sequence_type: # eg ; aa, nt
                        if process == "sketch":
                            if seqtype == "aa":
                                infile = get_gene_list_for_sketch(simdir, "aa")
                            else :
                                infile = get_gene_list_for_sketch(simdir, "nt")
                        else :
                            infile = get_clustering_fasta(simdir, seqtype)
                        for c_value in get_c_values_for_process(process, seqtype):
                            suffix = f"_st-{seqtype}" + (
                                f"_c-{c_value}" if c_value != DEFAULT_PARAMS["c"] else ""
                            )
                            jobinfo.append(
                                get_command_for_process(
                                    process,
                                    seqtype,
                                    infile,
                                    os.path.join(
                                        generaloutdir,
                                        "simulations",
                                        assembly,
                                        str(seed),
                                        process + suffix,
                                    ),
                                    args.threads,
                                    args.mem,
                                    args.softwaredir,
                                    c_value,
                                    gcbrepo=args.gcb_repo
                                )
                            )

    if not jobinfo:
        raise RuntimeError(
            "No clustering jobs were prepared; expected simulations under "
            f"{simulations_dir}/<assembly>/<seed>"
        )

    print("\n> Writing job commands file...")
    with open(os.path.join("./", COMMANDS_FILE), "w") as handle:
        for i, command in enumerate(jobinfo):
            # one job at a time in the run_benchmark.py file that is missing
            handle.write(f"{i}\t{command}\n")
    print("> Done!")

    # SLURM submission command
    print("\n> Launching job array...")
    tmpwrapcmd = EXEC_SCAFFOLD.format(
        executable=args.benchmark_runner,
        filepath=os.path.join(os.getcwd(), COMMANDS_FILE),
    )

    arraylogpath = os.path.join(args.outdir, "logs")
    if not os.path.isdir(arraylogpath) and not args.pretend:
        os.makedirs(arraylogpath)

    actualnjobs = int(args.max_simultaneous_cores / args.threads)
    arrayvals = f"0-{len(jobinfo) - 1}" + (
        f"%{actualnjobs}" if actualnjobs > 1 else "%1"
    )
    tmpcomm = SLURM_SCAFFOLD.format(
        nth=args.threads,
        timemax=args.time,
        memmax=args.mem,
        arrayvals=arrayvals,
        jobname=f"BenchmarkClustering_{timestamp}",
        logpath=arraylogpath,
        command=tmpwrapcmd,
        other="",
    )

    if args.pretend:
        print(f"You'd execute {tmpcomm}")
    else:
        print(f"Executing {tmpcomm}")
        os.system(tmpcomm)
        print(
            "\n> The temporal output folder has timestamp "
            f"{timestamp} and is:\n\t{timestamp}\nYou'll need it later!"
        )


def main():
    parser = argparse.ArgumentParser(
        usage="geneclusterbench-submit-clustering",
        description="Benchmark gene clustering software.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datapath", default=DEFAULT_DATAPATH)
    parser.add_argument("--seeds", "-s", default=DEFAULT_SEEDS)
    parser.add_argument("--outdir", "-o", default="./")
    parser.add_argument("--temp-outdir", "-to", default="/hps/nobackup/jlees/campan/tmp/")
    parser.add_argument("--threads", "-j", default=8, type=int)
    parser.add_argument("--time", "-t", default="1-12:00:00")
    parser.add_argument("--mem", "-m", default="48")
    parser.add_argument("--max-simultaneous-cores", "-M", default=2000, type=int)
    parser.add_argument("--preset-timestamp", "-P", default=-1, type=int)
    parser.add_argument("--pretend", "-p", action="store_true")
    parser.add_argument("--process", "-pr", default="cdhit,mmseqs2,diamond,panaroo,ppanggolin,panta,panx,sketch,embeddings")
    parser.add_argument("--sequence-type", "-st", default="nt,aa")
    parser.add_argument("--softwaredir", default=DEFAULT_SOFTWAREDIR)
    parser.add_argument("--benchmark-runner", default=DEFAULT_RUNNER)
    parser.add_argument("--gcb-repo", default=DEFAULT_GCB_REPO)
    args = parser.parse_args()

    args.process = args.process.strip().split(",")
    args.sequence_type = args.sequence_type.strip().split(",")

    if args.pretend:
        print("\n#===========# PRETENDING #===========#\n")

    if args.outdir in [".", "./"]:
        args.outdir = os.getcwd()
        print(f"> Changing output directory to full path ({args.outdir})")

    submit_clustering_jobs(args)


if __name__ == "__main__":
    main()
