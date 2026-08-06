#!/bin/bash

mkdir -p logs

seeds=(34 53144 40547 60207 21708 31001 54634 29492 6032 30354 5319 46118 1681 27347 14928 14557 62092 49444 25172 25913 31375 13478 14720 1274 11998 5455 56065 35787 28734 1894)

simulations_dir=/nfs/research/jlees/campan/data/clustering_benchmarking/2026_06_10_simsnowwithntandaasandgffs/simulations

for assembly_dir in "${simulations_dir}"/PROKKA_06122025__*; do
  assembly_name=$(basename "${assembly_dir}")

  echo "Starting assembly ${assembly_name}"

  for seed in "${seeds[@]}"; do

    srun --job-name=embedprots_${assembly_name}_${seed} \
         --cpus-per-task=8 \
         --mem=256G \
         --time=23:00:00 \
         --output=logs/embed_${assembly_name}_${seed}.out \
         --error=logs/embed_${assembly_name}_${seed}.err \
         python embedprots.py \
           --input-fasta "${assembly_dir}/${seed}"/*_for_clustering_aa.fasta \
           --out-dir /hps/nobackup/jlees/campan/embd/${assembly_name}/${seed} \
           --batch-size 32 \
           --nthreads 8 &

  done

  # Wait until all seeds for this assembly finish
  wait

  echo "Finished assembly ${assembly_name}"

done
