#!/bin/bash
#SBATCH --job-name=rag_generation
#SBATCH --output=results/logs/generation_eval_%j.out
#SBATCH --error=results/logs/generation_eval_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

cd $SLURM_SUBMIT_DIR
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multimodal_RAG

python scripts/evaluate.py \
  --benchmark data/benchmark/benchmark.json \
  --config configs/pipeline.yaml \
  --output results/generation_eval_70.json \
  --no-ragas \
  --max-examples 20
