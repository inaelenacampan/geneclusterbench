# geneclusterbench

`geneclusterbench` packages utilities for creating phylogenetically evolved
pangenome simulations and benchmarking gene clustering tools against
simulator truth labels.

The workflow covers:

1. Create simulated pangenomes from one annotated genome.
2. Submit many simulation jobs to Slurm.
3. Generate protein embeddings.
4. Cluster (via CD-HIT/MMseqs2 etc, or via embeddings).
5. Analyse clustering outputs and generate plots.


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
uv run python -m geneclusterbench.embedprots --help
uv run python -m geneclusterbench.cluster_embeddings --help
```

Alternatively, activate the UV-created environment and use `python -m` directly:

```bash
source .venv/bin/activate
python -m geneclusterbench.simulate_full_pangenome --help
```

## Default seed file

Batch scripts read seeds from `data/random_numbers.txt` by default: one
seed per line, 30 seeds. Pass `--seeds` to use a different seed file.

## External Tools

The Python environment includes the simulation and analysis dependencies,
but the clustering benchmark still expects external command-line tools and
cluster infrastructure:

- Slurm, for job submission.
- MMseqs2
- CD-HIT (protein clustering)
- CD-HIT-EST (nucleotide clustering)
- The external benchmark runner
- Diamond
- Panaroo
- Ppanggolin
- Panta
- PanX
- ProtT5
- sketchlib

**These are expected under hardcoded paths that are specific to the original
author's filesystem and cluster environment. Every user needs to update
these paths before running the pipeline on their own checkout.**

The benchmark sweeps `c` as sequence identity:

- CD-HIT receives this as `-c`, with `-n` chosen automatically from the
  sequence type and threshold. Nucleotide CD-HIT (`cd-hit-est`) skips
  thresholds below `0.8`.
- MMseqs2 receives this as `--min-seq-id`; MMseqs2 coverage is left at its default.

The software directory can be set with:

```bash
uv run python -m geneclusterbench.submit_gene_clustering --softwaredir /path/to/software
```

## Files

- `src/geneclusterbench/simulate_full_pangenome.py`: reads a GFF3 file with
  embedded FASTA, simulates gene gain/loss and mutation over a phylogeny,
  and writes simulated FASTA/GFF files plus truth and clustering input files.
- `src/geneclusterbench/submit_simulations.py`: Slurm launcher for running
  many pangenome simulations from a seed file.
- `src/geneclusterbench/submit_gene_clustering.py`: Slurm launcher for
  CD-HIT and MMseqs2 gene-clustering benchmarks over nucleotide and
  amino-acid clustering FASTAs.
- `src/geneclusterbench/analyse_gene_clustering.py`: parses CD-HIT and
  MMseqs2 outputs, compares clusters to simulator truth labels, computes
  clustering metrics, and plots results.
- `src/geneclusterbench/cluster_distance_file.py`: reads a dist file from
  sketchlib and does simple clustering on it.
- `src/geneclusterbench/embedprots.py`: generates protein embeddings.
- `src/geneclusterbench/cluster_embeddings.py`: clusters embeddings.
- `docs/simulation_to_analysis_workflow.md`: end-to-end workflow notes for
  starting from one genome, creating simulations, processing them on Slurm,
  and analysing outputs.

## Pipeline

For simulated data: simulate → submit simulations → generate embeddings →
cluster embeddings → analyse.

## Outputs

Analysis outputs are written as TSV files plus figures, reporting ARI
(Adjusted Rand Index), AMI (Adjusted Mutual Information), Dice coefficient,
and p-values for each clustering method/threshold combination.
