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

cd $SLURM_SUBMIT_DIR
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multimodal_RAG

python scripts/run_pruning_ablation.py
