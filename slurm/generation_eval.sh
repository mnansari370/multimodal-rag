#!/bin/bash
#SBATCH --job-name=rag_generation
#SBATCH --output=results/logs/generation_eval_%j.out
#SBATCH --error=results/logs/generation_eval_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# End-to-end generation evaluation using Claude via the Anthropic API.
#
# Runs retrieve → rerank → prune → generate on all benchmark examples,
# then computes token F1, citation accuracy, and efficiency metrics.
#
# RAGAS faithfulness evaluation is enabled — it calls the Claude API as
# an LLM judge. Make sure ANTHROPIC_API_KEY is set.
#
# GPU is needed for retrieval (dense embeddings) and reranking.

cd $SLURM_SUBMIT_DIR

# Load local environment variables if .env exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set"
  exit 1
fi


source ~/miniconda3/etc/profile.d/conda.sh
conda activate multimodal_RAG

mkdir -p results/logs

echo "=== End-to-end generation evaluation ==="
python scripts/evaluate.py \
    --benchmark data/benchmark/benchmark.json \
    --config configs/pipeline_claude.yaml \
    --output results/generation_eval_full.json

echo "=== Human evaluation (interactive — skip in batch) ==="
# Uncomment to run interactive human eval on 25 examples:
# python scripts/human_eval.py \
#     --benchmark data/benchmark/benchmark.json \
#     --config configs/pipeline_claude.yaml \
#     --num-examples 25 \
#     --output results/human_eval.json

echo "=== Failure analysis ==="
python scripts/analyze_failures.py \
    --benchmark data/benchmark/benchmark.json \
    --output results/failure_analysis.json \
    --top-k 10

echo "=== Updating results tables ==="
python scripts/make_results_tables.py

echo "Generation evaluation complete."
