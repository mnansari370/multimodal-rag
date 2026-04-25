#!/bin/bash
#SBATCH --job-name=embed_corpus
#SBATCH --output=results/logs/embed_corpus_%j.out
#SBATCH --error=results/logs/embed_corpus_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Build FAISS dense index over the chunked corpus
# Requires GPU for fast embedding generation

cd $SLURM_SUBMIT_DIR

source activate DL_env

CHUNKS_FILE=${1:-data/processed/chunks_heading.jsonl}
echo "=== Building dense embeddings for: $CHUNKS_FILE ==="

python - <<EOF
import json
from src.retrieval import HybridRetriever

chunks = []
with open("$CHUNKS_FILE") as f:
    for line in f:
        chunks.append(json.loads(line.strip()))

print(f"Loaded {len(chunks)} chunks")

retriever = HybridRetriever()
retriever.build(chunks)
retriever.save(
    bm25_path="data/embeddings/bm25.pkl",
    faiss_path="data/embeddings/dense.faiss",
    chunks_path="data/embeddings/chunks.jsonl",
)
print("Embeddings saved to data/embeddings/")
EOF

echo "Embedding complete."
