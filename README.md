# geneclusterbench

`geneclusterbench` packages utilities for creating phylogenetically evolved pangenome simulations and benchmarking gene clustering tools against simulator truth labels.

The current workflow covers:

1. Create simulated pangenomes from one annotated genome.
2. Submit many simulation jobs to Slurm.
3. Submit gene-clustering jobs to Slurm.
4. Analyse gene-clustering outputs and generate plots.

Benchmarking of clusters of gene clusters is not implemented yet.

## Setup

Create the UV environment from the repository root:

```bash
uv sync
```

Run modules through UV:

```bash
uv run python -m geneclusterbench.simulate_full_pangenome --help
uv run python -m geneclusterbench.submit_simulations --help
uv run python -m geneclusterbench.submit_gene_clustering --help
uv run python -m geneclusterbench.analyse_gene_clustering --help
uv run python -m geneclusterbench.cluster_distance_file --help
```

Alternatively, activate the UV-created environment and use `python -m` directly:

```bash
source .venv/bin/activate
python -m geneclusterbench.simulate_full_pangenome --help
```

## Default seed file

Batch scripts read seeds from `data/random_numbers.txt` by default. Pass `--seeds` to use a different seed file.

## External Tools

The Python environment includes the simulation and analysis dependencies, but the clustering benchmark still expects external command-line tools and cluster infrastructure:

- Slurm, for job submission.
- MMseqs2, currently expected under `softwaredir/mmseqs2/MMseqs2/build/bin/mmseqs`.
- CD-HIT protein clustering, currently expected under `softwaredir/cdhit/cdhit/cd-hit`.
- CD-HIT nucleotide clustering, currently expected under `softwaredir/cdhit/cdhit/cd-hit-est`.
- The external benchmark runner, currently defaulting to `/hps/software/users/jlees/vrbouza/projects/assembler_development/benchmarking/run_benchmark.py`.

Before running this from a cloned or copied checkout, update the local hardcoded paths for your filesystem and cluster environment.

The benchmark sweeps `c` as sequence identity. CD-HIT receives this as `-c`, with `-n` chosen from the sequence type and threshold; nucleotide CD-HIT uses `cd-hit-est`, so thresholds below `0.8` are skipped. MMseqs2 receives it as `--min-seq-id`; MMseqs2 coverage is left at its default.

The default software directory is:

```python
softwaredir = "/hps/software/users/jlees/vrbouza/projects/clustering_benchmark/software"
```

You can override it with:

```bash
uv run python -m geneclusterbench.submit_gene_clustering --softwaredir /path/to/software
```

## Files

- `src/geneclusterbench/simulate_full_pangenome.py`: reads a GFF3 file with embedded FASTA, simulates gene gain/loss and mutation over a phylogeny, and writes simulated FASTA/GFF files plus truth and clustering input files.
- `src/geneclusterbench/submit_simulations.py`: Slurm launcher for running many pangenome simulations from a seed file.
- `src/geneclusterbench/submit_gene_clustering.py`: Slurm launcher for CD-HIT and MMseqs2 gene-clustering benchmarks over nucleotide and amino-acid clustering FASTAs.
- `src/geneclusterbench/analyse_gene_clustering.py`: parses CD-HIT and MMseqs2 outputs, compares clusters to simulator truth labels, computes clustering metrics, and plots results.
- `src/geneclusterbench/cluster_distance_file.py`: reads a dist file from sketchlib and does simple clustering on it.
- `docs/simulation_to_analysis_workflow.md`: end-to-end workflow notes for starting from one genome, creating simulations, processing them on Slurm, and analysing outputs.
