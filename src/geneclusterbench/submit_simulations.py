import argparse
import os
import time
from pathlib import Path
import glob
import re
import csv
import statistics

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

RATE_FILENAME_RE = re.compile(
    r"sim_gr_(?P<gain>[\-0-9.eE]+)_lr_(?P<loss>[\-0-9.eE]+)_mu_(?P<mut>[\-0-9.eE]+)_presence_absence\.csv$"
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

def count_genes_in_gff(gff_path):
    n_cds = 0
    with open(gff_path, "r") as f:
        for line in f:
            if line.startswith("##FASTA"):
                break
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3 and fields[2] == "CDS":
                n_cds += 1
    return n_cds

def find_presence_absence_file(seed_dir):
    matches = glob.glob(os.path.join(seed_dir, "*_presence_absence.csv"))
    if not matches:
        return None
    matches.sort()
    return matches[0]
 
 
def read_presence_absence_stats(pa_path):
    with open(pa_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        iso_columns = header[1:]
        counts_per_isolate = [0] * len(iso_columns)
        total_genes = 0
        for row in reader:
            if not row:
                continue
            total_genes += 1
            for i, val in enumerate(row[1:]):
                if val.strip() == "1":
                    counts_per_isolate[i] += 1
 
    mean_genes_per_isolate = (
        statistics.mean(counts_per_isolate) if counts_per_isolate else 0.0
    )
    return total_genes, mean_genes_per_isolate
 
 
def collect_seed_metadata(assembly_dir, seeds):
    per_seed = []
    missing = []
 
    for seed in seeds:
        seed_dir = os.path.join(assembly_dir, str(seed))
        pa_path = find_presence_absence_file(seed_dir) if os.path.isdir(seed_dir) else None
 
        if pa_path is None:
            missing.append(seed)
            continue
 
        rate_match = RATE_FILENAME_RE.search(os.path.basename(pa_path))
        gain_rate = rate_match.group("gain") if rate_match else "unknown"
        loss_rate = rate_match.group("loss") if rate_match else "unknown"
        mutation_rate = rate_match.group("mut") if rate_match else "unknown"
 
        total_genes, mean_genes_per_isolate = read_presence_absence_stats(pa_path)
 
        per_seed.append({
            "seed": seed,
            "gain_rate": gain_rate,
            "loss_rate": loss_rate,
            "mutation_rate": mutation_rate,
            "total_genes": total_genes,
            "mean_genes_per_isolate": mean_genes_per_isolate,
        })
 
    return per_seed, missing
 
 
def create_meta_data(metadata_path, gff, outdir_sims, assembly_name, seeds):
 
    original_gene_count = count_genes_in_gff(gff)
    assembly_dir = os.path.join(outdir_sims, "simulations", assembly_name)
 
    per_seed, missing = collect_seed_metadata(assembly_dir, seeds)
 
    lines = []
    lines.append("Pangenome simulation metadata")
    lines.append("=" * 30)
    lines.append(f"Original GFF: {gff}")
    lines.append(f"Number of genes in original GFF (CDS count): {original_gene_count}")
    lines.append(f"Assembly: {assembly_name}")
    lines.append(f"Seeds requested: {len(seeds)}")
    lines.append(f"Seeds with completed output: {len(per_seed)}")
    if missing:
        lines.append(f"Seeds missing/incomplete: {', '.join(str(s) for s in missing)}")
    lines.append("")
    lines.append("Per-seed results")
    lines.append("-" * 30)
 
    for record in per_seed:
        lines.append(f"Seed {record['seed']}")
        lines.append(f"  gain_rate (adjusted for pop size): {record['gain_rate']}")
        lines.append(f"  loss_rate (adjusted for pop size): {record['loss_rate']}")
        lines.append(f"  mutation_rate (adjusted for pop size): {record['mutation_rate']}")
        lines.append(f"  total genes in simulated pangenome: {record['total_genes']}")
        lines.append(f"  mean genes per isolate: {record['mean_genes_per_isolate']:.2f}")
        lines.append("")
 
    lines.append("Summary")
    lines.append("-" * 30)
    if per_seed:
        means = [r["mean_genes_per_isolate"] for r in per_seed]
        overall_mean = statistics.mean(means)
        lines.append(f"Mean genes per isolate across all seeds: {overall_mean:.2f}")
        if len(means) > 1:
            lines.append(f"Stdev of mean genes per isolate across seeds: {statistics.stdev(means):.2f}")
    else:
        lines.append("No completed seeds found yet; nothing to summarise.")
 
    report = "\n".join(lines) + "\n"
 
    with open(metadata_path, "w") as f:
        f.write(report)
 
    print(report)
    return metadata_path


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
    parser.add_argument(
        "--collect-metadata",
        action="store_true",
        help=(
            "Instead of submitting jobs, scan the output of a previous run "
            "and write a metadata report (original GFF gene count, "
            "gain/loss/mutation rates used, and mean genes per isolate per "
            "seed)."
        ),
    )
    parser.add_argument(
        "--metadata-out",
        default=None,
        help=(
            "Path for the metadata report (used with --collect-metadata). "
            "Defaults to '<outdir-sims>/simulations/<assembly-name>/metadata.txt'"
        ),
    )
    args = parser.parse_args()

    seedsfile = args.seeds
    randomnumbers = load_seeds(seedsfile)
    print(randomnumbers)

    assembly_name = args.assembly_name or get_assembly_name_from_gff(args.gff)
    mainoutputfolder = os.path.join(args.outdir_sims, "simulations")
    logdir = os.path.join(args.outdir_sims, "logs", "simulations")

    if args.collect_metadata:
        metadata_path = args.metadata_out or os.path.join(
            mainoutputfolder, assembly_name, "metadata.txt"
        )
        create_meta_data(
            metadata_path=metadata_path,
            gff=args.gff,
            outdir_sims=args.outdir_sims,
            assembly_name=assembly_name,
            seeds=randomnumbers,
        )
        return

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
