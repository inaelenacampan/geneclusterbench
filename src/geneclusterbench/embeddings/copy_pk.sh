#!/usr/bin/env bash
set -uo pipefail

SOURCE_DIR="/hps/nobackup/jlees/campan/embd"
DEST_DIR="/nfs/research/jlees/campan/data/clustering_benchmarking/2026_06_10_simsnowwithntandaasandgffs/simulations"

SEEDS=(
    34 53144 40547 60207 21708 31001 54634 29492 6032 30354
    5319 46118 1681 27347 14928 14557 62092 49444 25172 25913
    31375 13478 14720 1274 11998 5455 56065 35787 28734 1894
)

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "ERROR: Source directory does not exist: $SOURCE_DIR" >&2
    exit 1
fi

# Iterate over every PROKKA_* directory in the source (discovered automatically)
shopt -s nullglob
prokka_dirs=("$SOURCE_DIR"/PROKKA_*/)
shopt -u nullglob

if [[ ${#prokka_dirs[@]} -eq 0 ]]; then
    echo "ERROR: No PROKKA_* directories found in $SOURCE_DIR" >&2
    exit 1
fi

for prokka_path in "${prokka_dirs[@]}"; do
    # Strip trailing slash and get just the directory name
    prokka_path="${prokka_path%/}"
    prokka_name="$(basename "$prokka_path")"

    dest_prokka_dir="$DEST_DIR/$prokka_name"

    if [[ ! -d "$dest_prokka_dir" ]]; then
        echo "WARNING: Destination PROKKA directory does not exist, skipping all seeds for it: $dest_prokka_dir" >&2
        continue
    fi

    for seed in "${SEEDS[@]}"; do
        src_seed_dir="$prokka_path/$seed"
        dest_seed_dir="$dest_prokka_dir/$seed"

        if [[ ! -d "$src_seed_dir" ]]; then
            echo "WARNING: Source seed directory missing, skipping: $src_seed_dir" >&2
            continue
        fi

        # Find .pk files in the source seed directory
        shopt -s nullglob
        pk_files=("$src_seed_dir"/*.pk)
        shopt -u nullglob

        if [[ ${#pk_files[@]} -eq 0 ]]; then
            echo "WARNING: No .pk file found in: $src_seed_dir" >&2
            continue
        fi

        if [[ ${#pk_files[@]} -gt 1 ]]; then
            echo "WARNING: More than one .pk file found in: $src_seed_dir (copying all of them)" >&2
        fi

        # Create destination seed directory if it doesn't exist
        if [[ ! -d "$dest_seed_dir" ]]; then
            mkdir -p "$dest_seed_dir"
        fi

        for pk_file in "${pk_files[@]}"; do
            filename="$(basename "$pk_file")"
            dest_file="$dest_seed_dir/$filename"

            if cp -- "$pk_file" "$dest_file"; then
                echo "Copied: $pk_file -> $dest_file"
            else
                echo "WARNING: Failed to copy: $pk_file -> $dest_file" >&2
            fi
        done
    done
done

echo "Done."
