#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="/hps/nobackup/jlees/campan/embd/PROKKA_06122025__gr_1e-12_lr_1e-12_mu_1e-14"
DST_BASE="/nfs/research/jlees/campan/data/clustering_benchmarking/2026_06_10_simsnowwithntandaasandgffs/simulations/PROKKA_06122025__gr_1e-12_lr_1e-12_mu_1e-14"

for seed_dir in "$SRC_BASE"/*/; do
    seed=$(basename "$seed_dir")

    # Find the .pk file in this seed directory (non-recursive)
    pk_file=$(find "$seed_dir" -maxdepth 1 -type f -name "*.pk" -print -quit)

    if [[ -z "$pk_file" ]]; then
        echo "WARNING: No .pk file found in seed directory '$seed'" >&2
        continue
    fi

    dest_dir="$DST_BASE/$seed"
    mkdir -p "$dest_dir"

    cp -- "$pk_file" "$dest_dir"/

    echo "Copied '$pk_file' -> '$dest_dir/$(basename "$pk_file")'"
done
