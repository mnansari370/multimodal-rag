#!/bin/bash
#SBATCH --job-name=rag_retrieval
#SBATCH --output=results/logs/retrieval_eval_%j.out
#SBATCH --error=results/logs/retrieval_eval_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Retriever ablation study.
# Compares BM25, dense, hybrid, and hybrid+reranker retrieval quality.
# Produces the retrieval section of the main results table.
#
# GPU is needed for the dense embedding queries during retrieval.

cd $SLURM_SUBMIT_DIR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multimodal_RAG

mkdir -p results/logs

echo "=== Retriever ablation ==="
python scripts/run_retriever_ablation.py \
    --benchmark data/benchmark/benchmark.json \
    --output results/retriever_ablation.json

echo "=== Chunking comparison ==="
python scripts/run_chunking_comparison.py

echo "=== Building results tables ==="
python scripts/make_results_tables.py

echo "Retrieval evaluation complete."
