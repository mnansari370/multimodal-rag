#!/bin/bash
#SBATCH --job-name=rag_embed
#SBATCH --output=results/logs/embed_corpus_%j.out
#SBATCH --error=results/logs/embed_corpus_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Build FAISS dense embeddings and BM25 index over the chunked corpus.
# Requires GPU for fast batch encoding with the bi-encoder model.
#
# By default uses heading-based chunks. Pass a different chunks file
# via the first argument to use fixed-size chunks:
#   sbatch slurm/embed_corpus.sh data/processed/chunks_fixed.jsonl

cd $SLURM_SUBMIT_DIR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multimodal_RAG

mkdir -p results/logs data/embeddings

CHUNKS_FILE=${1:-data/processed/chunks_heading.jsonl}
echo "=== Building retrieval indexes from: $CHUNKS_FILE ==="

python scripts/build_index.py \
    --chunks "$CHUNKS_FILE" \
    --bm25-out data/embeddings/bm25.pkl \
    --faiss-out data/embeddings/dense.faiss \
    --chunks-out data/embeddings/chunks.jsonl

echo "=== Index statistics ==="
python scripts/validate_data.py

echo "Embedding complete."
