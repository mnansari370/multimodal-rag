#!/bin/bash
#SBATCH --job-name=rag_pruning
#SBATCH --output=results/logs/pruning_ablation_%j.out
#SBATCH --error=results/logs/pruning_ablation_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Context pruning ablation study (Section 6.2 of the project report).
# Measures how different pruning strategies affect prompt token count
# and retrieval latency. Does NOT call the generation API to keep cost down.
#
# Produces results/pruning_ablation.json which feeds into make_results_tables.py.

cd $SLURM_SUBMIT_DIR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multimodal_RAG

mkdir -p results/logs

echo "=== Context pruning ablation ==="
python scripts/run_pruning_ablation.py \
    --benchmark data/benchmark/benchmark.json \
    --output results/pruning_ablation.json

echo "=== Updating results tables ==="
python scripts/make_results_tables.py

echo "Pruning ablation complete."
