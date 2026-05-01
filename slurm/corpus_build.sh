#!/bin/bash
#SBATCH --job-name=rag_corpus
#SBATCH --output=results/logs/corpus_build_%j.out
#SBATCH --error=results/logs/corpus_build_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch

# Download and clean PyTorch documentation corpus, then chunk it.
# CPU-only task — no GPU needed.
#
# Steps:
#   1. Download all PyTorch stable docs from the sitemap
#   2. Clean and normalize each page
#   3. Chunk using heading-based strategy (for production)
#   4. Chunk using fixed-size strategy (for comparison ablation)

cd $SLURM_SUBMIT_DIR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multimodal_RAG

mkdir -p results/logs data/raw data/processed

echo "=== Downloading PyTorch documentation ==="
python scripts/ingest.py \
    --raw-dir data/raw \
    --processed-dir data/processed \
    --delay 0.3

echo ""
echo "=== Corpus statistics ==="
python scripts/validate_data.py

echo "Corpus build complete."
