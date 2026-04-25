#!/bin/bash
#SBATCH --job-name=reranking_eval
#SBATCH --output=results/logs/reranking_eval_%j.out
#SBATCH --error=results/logs/reranking_eval_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Evaluate the quality gain from adding reranking on top of hybrid retrieval
# Measures retrieval metrics before and after reranking to show the improvement

cd $SLURM_SUBMIT_DIR

source activate DL_env

BENCHMARK=${1:-data/benchmark/benchmark.json}
OUTPUT=results/reranking_comparison.json

python - <<EOF
import json
from src.retrieval import HybridRetriever
from src.reranking import Reranker
from src.evaluation import evaluate_retrieval, print_retrieval_results

with open("$BENCHMARK") as f:
    benchmark = json.load(f)

retriever = HybridRetriever()
retriever.load()

reranker = Reranker()
results_all = {}

print("=== Without reranking (hybrid only) ===")
fn_no_rerank = lambda q: retriever.search(q, top_k=20)
metrics_no_rerank = evaluate_retrieval(benchmark, fn_no_rerank)
print_retrieval_results(metrics_no_rerank, label="hybrid, no rerank")
results_all["hybrid_no_rerank"] = metrics_no_rerank

print("\n=== With reranking ===")
def fn_with_rerank(q):
    candidates = retriever.search(q, top_k=50)
    return reranker.rerank(q, candidates, top_k=20)

metrics_rerank = evaluate_retrieval(benchmark, fn_with_rerank)
print_retrieval_results(metrics_rerank, label="hybrid + rerank")
results_all["hybrid_with_rerank"] = metrics_rerank

print("\n=== Delta (rerank - no_rerank) ===")
for metric in metrics_rerank:
    delta = metrics_rerank[metric] - metrics_no_rerank.get(metric, 0)
    print(f"  {metric:<20} {delta:+.4f}")

with open("$OUTPUT", "w") as f:
    json.dump(results_all, f, indent=2)

print(f"\nSaved to $OUTPUT")
EOF

echo "Reranking evaluation complete."
