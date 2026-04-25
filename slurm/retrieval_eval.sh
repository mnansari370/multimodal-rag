#!/bin/bash
#SBATCH --job-name=retrieval_eval
#SBATCH --output=results/logs/retrieval_eval_%j.out
#SBATCH --error=results/logs/retrieval_eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Run retrieval evaluation (BM25 vs. dense vs. hybrid)
# and chunking strategy comparison (heading vs. fixed-size)

cd $SLURM_SUBMIT_DIR

source activate DL_env

BENCHMARK=${1:-data/benchmark/benchmark.json}
OUTPUT=results/retrieval_results.json

python - <<EOF
import json
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from src.evaluation import evaluate_retrieval, print_retrieval_results

with open("$BENCHMARK") as f:
    benchmark = json.load(f)

results_all = {}

for retriever_type in ["bm25", "dense", "hybrid"]:
    print(f"\n=== Evaluating {retriever_type} retriever ===")

    if retriever_type == "bm25":
        r = BM25Retriever()
        r.load("data/embeddings/bm25.pkl")
        fn = lambda q: r.search(q, top_k=20)
    elif retriever_type == "dense":
        r = DenseRetriever()
        r.load("data/embeddings/dense.faiss", "data/embeddings/chunks.jsonl")
        fn = lambda q: r.search(q, top_k=20)
    else:
        r = HybridRetriever()
        r.load()
        fn = lambda q: r.search(q, top_k=20)

    metrics = evaluate_retrieval(benchmark, fn)
    print_retrieval_results(metrics, label=retriever_type)
    results_all[retriever_type] = metrics

with open("$OUTPUT", "w") as f:
    json.dump(results_all, f, indent=2)

print(f"\nSaved to $OUTPUT")
EOF

echo "Retrieval evaluation complete."
