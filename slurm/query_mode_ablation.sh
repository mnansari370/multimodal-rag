#!/bin/bash
#SBATCH --job-name=rag_query_modes
#SBATCH --output=results/logs/query_mode_ablation_%j.out
#SBATCH --error=results/logs/query_mode_ablation_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

cd $SLURM_SUBMIT_DIR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multimodal_RAG

export TOKENIZERS_PARALLELISM=false

mkdir -p results/logs results/tables

echo "=== GPU check ==="
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo ""
echo "=== Query-mode ablation: hybrid without reranker ==="
python scripts/run_query_mode_ablation.py \
  --benchmark data/benchmark/benchmark.json \
  --vlp-outputs results/vlp_outputs.json \
  --output results/query_mode_ablation.json

echo ""
echo "=== Query-mode ablation: hybrid + reranker ==="
python scripts/run_query_mode_ablation.py \
  --benchmark data/benchmark/benchmark.json \
  --vlp-outputs results/vlp_outputs.json \
  --output results/query_mode_ablation_reranker.json \
  --use-reranker

echo ""
echo "=== Results ==="
python -m json.tool results/query_mode_ablation.json
python -m json.tool results/query_mode_ablation_reranker.json

echo "Query-mode ablation complete."
