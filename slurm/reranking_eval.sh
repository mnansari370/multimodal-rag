#!/bin/bash
#SBATCH --job-name=rag_reranking
#SBATCH --output=results/logs/reranking_eval_%j.out
#SBATCH --error=results/logs/reranking_eval_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Evaluate the quality gain from adding cross-encoder reranking on top of
# hybrid retrieval. This contributes two rows to the main results table:
#   - hybrid only
#   - hybrid + reranker
#
# The reranker runs on CPU for small batches but benefits from GPU when
# scoring many candidates across the full benchmark.

cd $SLURM_SUBMIT_DIR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multimodal_RAG

mkdir -p results/logs

echo "=== Reranking evaluation ==="
python scripts/run_retriever_ablation.py \
    --benchmark data/benchmark/benchmark.json \
    --output results/retriever_ablation.json

echo "Reranking evaluation complete."
