#!/bin/bash
#SBATCH --job-name=rag_vlp
#SBATCH --output=results/logs/vlp_run_%j.out
#SBATCH --error=results/logs/vlp_run_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch

# Run the vision-language parser over all benchmark screenshots.
# Saves one structured output per example to results/vlp_outputs.json.
#
# Using the Anthropic API backend (Claude Haiku) — no GPU needed.
# If you prefer to run InternVL2 locally, change --backend to internvl2
# and add --gres=gpu:1 --partition=gpu to the SBATCH headers above.
#
# Make sure ANTHROPIC_API_KEY is set in your environment before submitting.

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

echo "=== Running VLP batch over benchmark screenshots ==="
python scripts/run_vlp_batch.py \
    --benchmark data/benchmark/benchmark.json \
    --output results/vlp_outputs.json \
    --backend anthropic \
    --resume

echo "VLP batch complete."
