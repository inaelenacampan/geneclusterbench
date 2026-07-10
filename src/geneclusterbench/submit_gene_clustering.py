# import libraries

import argparse
import os
import time
from pathlib import Path

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
    "{execexec} -fn {workdir} -sl {species} -t {ncores} -st 1 3 4 5 -dmi {dmi} && "
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


def get_command_for_process(proc, seqtype, infile, outfolder, nthreads, maxmem, softwaredir, c=0.9):
    
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

        # panX expects input GenBank files under <workdir>/data/<species>/input_GenBank,
        # and takes that species label via -sl. Since these are simulated genomes
        # with no real species name, use a fixed placeholder that must stay
        # consistent with wherever the analysis script looks for panX output.
        species = "simulated_species"
        gbkdir = os.path.join(outfolder, "data", species, "input_GenBank")

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

def get_sim_iso_gffs(simdir):
    matches = [
        os.path.join(simdir, el) for el in os.listdir(simdir)
        if el.endswith(".gff")
    ]
    if not matches:
        raise RuntimeError(f"No GFF files found in {simdir}")
    return " ".join(matches)  # e.g. "path/iso_0.gff path/iso_1.gff ..."

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
                            )
                        )
                else:
                    for seqtype in args.sequence_type: # eg ; aa, nt
                        
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
    parser.add_argument("--threads", "-j", default=4, type=int)
    parser.add_argument("--time", "-t", default="1-12:00:00")
    parser.add_argument("--mem", "-m", default="48")
    parser.add_argument("--max-simultaneous-cores", "-M", default=2000, type=int)
    parser.add_argument("--preset-timestamp", "-P", default=-1, type=int)
    parser.add_argument("--pretend", "-p", action="store_true")
    parser.add_argument("--process", "-pr", default="cdhit,mmseqs2,diamond,panaroo,ppanggolin,panta,panx")
    parser.add_argument("--sequence-type", "-st", default="nt,aa")
    parser.add_argument("--softwaredir", default=DEFAULT_SOFTWAREDIR)
    parser.add_argument("--benchmark-runner", default=DEFAULT_RUNNER)
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
