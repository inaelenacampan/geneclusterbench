import argparse
import os
import time
from pathlib import Path


DEFAULT_OUTDIRSIMS = (
    "/nfs/research/jlees/campan/data/clustering_benchmarking/"
    "2026_06_10_simsnowwithntandaasandgffs"
)
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_EXEC_PATH = str(PACKAGE_DIR / "simulate_full_pangenome.py")
DEFAULT_PY3ENV = str(PROJECT_ROOT / ".venv" / "bin" / "activate")
DEFAULT_SEEDS = str(PROJECT_ROOT / "data" / "random_numbers.txt")

DEFAULT_GFF = (
    "/nfs/research/jlees/campan/data/clustering_benchmarking/"
    "2026_06_10_simsnowwithntandaasandgffs/MSdataset/6925_1#61/PROKKA_06122025.gff"
)

GENERATION_SCAFFOLD = '. "{env}" && python3 "{execexec}" -g "{inputgff}" -o "{outputpath}" -s "{seed}"'
SLURM_SCAFFOLD = (
    "sbatch -c 1 -t {timemax} --mem {memmax}G -J {jobname} "
    "-e {logpath}/log.%A.%a.%x.err -o {logpath}/log.%A.%a.%x.out "
    "--wrap '{command}' {other}"
)


def load_seeds(seedsfile):
    seeds = []
    with open(seedsfile, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                seeds.append(int(stripped))
    return seeds


def get_assembly_name_from_gff(gff_path):
    return Path(gff_path).stem


def main():
    parser = argparse.ArgumentParser(
        description="Submit pangenome simulation jobs to Slurm."
    )
    parser.add_argument("--outdir-sims", default=DEFAULT_OUTDIRSIMS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--simulator", default=DEFAULT_EXEC_PATH)
    parser.add_argument("--python-env", default=DEFAULT_PY3ENV)
    parser.add_argument("--gff", default=DEFAULT_GFF)
    parser.add_argument("--time", dest="timemax", default="1-00:00:00")
    parser.add_argument("--mem", dest="memmax", default=6, type=int)
    parser.add_argument("--job-name", default="pangenomesims")
    parser.add_argument(
        "--assembly-name",
        default=None,
        help="Assembly folder name; defaults to the input GFF basename without extension",
    )
    parser.add_argument("--pretend", action="store_true")
    args = parser.parse_args()

    seedsfile = args.seeds
    randomnumbers = load_seeds(seedsfile)
    print(randomnumbers)

    assembly_name = args.assembly_name or get_assembly_name_from_gff(args.gff)
    mainoutputfolder = os.path.join(args.outdir_sims, "simulations")
    logdir = os.path.join(args.outdir_sims, "logs", "simulations")
    if not args.pretend:
        os.makedirs(logdir, exist_ok=True)
    timestamp = time.time()

    for seed in randomnumbers:
        tmpoutpath = os.path.join(mainoutputfolder, assembly_name, str(seed))
        if not os.path.isdir(tmpoutpath) and not args.pretend:
            os.makedirs(tmpoutpath)

        tmpcomm = SLURM_SCAFFOLD.format(
            timemax=args.timemax,
            memmax=args.memmax,
            jobname=args.job_name,
            logpath=logdir,
            command=GENERATION_SCAFFOLD.format(
                execexec=args.simulator,
                env=args.python_env,
                inputgff=args.gff,
                outputpath=tmpoutpath,
                seed=seed,
            ),
            other="",
        )

        if args.pretend:
            print(f"You'd execute:\n\t{tmpcomm}\n")
        else:
            print(f"Executing\n\t{tmpcomm}")
            os.system(tmpcomm)


if __name__ == "__main__":
    main()
