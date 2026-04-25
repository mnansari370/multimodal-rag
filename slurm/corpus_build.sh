#!/bin/bash
#SBATCH --job-name=corpus_build
#SBATCH --output=results/logs/corpus_build_%j.out
#SBATCH --error=results/logs/corpus_build_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch

# Download and clean PyTorch documentation corpus
# This is a CPU-only task — no GPU needed

cd $SLURM_SUBMIT_DIR

source activate DL_env

echo "=== Downloading PyTorch documentation ==="
python -m src.ingestion.downloader \
    --output-dir data/raw \
    --delay 0.3

echo "=== Cleaning raw pages ==="
python -m src.ingestion.cleaner \
    --raw-dir data/raw \
    --output-dir data/processed

echo "=== Chunking (heading-based) ==="
python -m src.chunking.chunker \
    --processed-dir data/processed \
    --output-file data/processed/chunks_heading.jsonl \
    --strategy heading

echo "=== Chunking (fixed-size with overlap) ==="
python -m src.chunking.chunker \
    --processed-dir data/processed \
    --output-file data/processed/chunks_fixed.jsonl \
    --strategy fixed

echo "Corpus build complete."
