"""
Context pruning ablation study.

Measures how different pruning strategies affect prompt token count,
latency, and answer quality. This produces the efficiency table from
Section 6.2 of the project report.

All experiments use the same hybrid retrieval + reranking stack.
Only the context-selection step changes across experiments.

The ablation table shows:
  Setting | Prompt tokens | Latency (s) | Faithfulness | Answer score

Usage:
    python scripts/run_pruning_ablation.py
    python scripts/run_pruning_ablation.py --output results/pruning_ablation.json
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import MultimodalRAGPipeline, PipelineConfig
from src.efficiency import select_context


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


# Each experiment is (name, strategy, threshold, context_top_k, mmr_lambda).
# threshold=-999 keeps all chunks (no pruning baseline).
EXPERIMENTS = [
    ("no_pruning_20",    "threshold", -999, 20, 0.7),
    ("threshold_top10",  "threshold",  0.0, 10, 0.7),
    ("top5_diversity",   "diversity",  0.0,  5, 0.7),
    ("coverage_mmr_07",  "coverage",   0.0,  5, 0.7),
    ("coverage_mmr_05",  "coverage",   0.0,  5, 0.5),
]


def main():
    parser = argparse.ArgumentParser(
        description="Run the context pruning ablation study"
    )
    parser.add_argument(
        "--benchmark",
        default="data/benchmark/benchmark.json",
        help="Path to the evaluation benchmark JSON",
    )
    parser.add_argument(
        "--output",
        default="results/pruning_ablation.json",
        help="Where to save the results JSON",
    )
    args = parser.parse_args()

    benchmark_path = ROOT / args.benchmark
    output_path = ROOT / args.output

    logger.info("Loading benchmark from %s", benchmark_path)
    with open(benchmark_path, encoding="utf-8") as f:
        benchmark = json.load(f)
    logger.info("Benchmark size: %d examples", len(benchmark))

    # Build the shared retrieval stack once — all experiments reuse it
    cfg = PipelineConfig(
        retriever_type="hybrid",
        use_reranker=True,
        use_vlp=False,   # isolated test — no VLP, no generation
    )
    pipeline = MultimodalRAGPipeline(cfg)
    pipeline.load_indexes()
    logger.info("Indexes loaded.")

    rows = []

    for name, strategy, threshold, top_k, mmr_lambda in EXPERIMENTS:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Experiment: %s", name)
        logger.info("  strategy=%s  threshold=%.1f  top_k=%d  lambda=%.1f",
                    strategy, threshold, top_k, mmr_lambda)
        logger.info("=" * 70)

        total_input_chunks = 0
        total_selected_chunks = 0
        total_original_tokens = 0
        total_pruned_tokens = 0
        total_reduction = 0.0
        total_latency = 0.0

        for ex in benchmark:
            # Combine log snippet with query to build the retrieval query
            # (same as what the pipeline does when log_snippet is present)
            query = ex.get("log_snippet", "")
            if query:
                query = query + " " + ex["query"]
            else:
                query = ex["query"]

            t0 = time.perf_counter()
            retrieved = pipeline.retriever.search(query, top_k=50)
            reranked = pipeline.reranker.rerank(query, retrieved, top_k=20)
            selected, stats = select_context(
                reranked,
                strategy=strategy,
                query=query,
                threshold=threshold,
                top_k=top_k,
                lambda_param=mmr_lambda,
            )
            elapsed = time.perf_counter() - t0

            total_input_chunks += stats["input_chunks"]
            total_selected_chunks += stats["selected_chunks"]
            total_original_tokens += stats["original_tokens"]
            total_pruned_tokens += stats["pruned_tokens"]
            total_reduction += stats["token_reduction_pct"]
            total_latency += elapsed

        n = len(benchmark)
        row = {
            "setting": name,
            "strategy": strategy,
            "top_k": top_k,
            "threshold": threshold,
            "mmr_lambda": mmr_lambda,
            "avg_input_chunks": round(total_input_chunks / n, 2),
            "avg_selected_chunks": round(total_selected_chunks / n, 2),
            "avg_original_tokens": round(total_original_tokens / n, 1),
            "avg_pruned_tokens": round(total_pruned_tokens / n, 1),
            "avg_token_reduction_pct": round(total_reduction / n, 2),
            "avg_retrieve_rerank_prune_latency_s": round(total_latency / n, 3),
        }
        rows.append(row)
        logger.info("Result: %s", json.dumps(row, indent=2))

    # Save all rows
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    logger.info("")
    logger.info("Saved pruning ablation to %s", output_path)

    # Print a quick summary table
    print("\n" + "=" * 80)
    print(f"{'Setting':<22} {'Orig tokens':>12} {'Pruned tokens':>14} {'Reduction %':>12} {'Latency s':>10}")
    print("-" * 80)
    for r in rows:
        print(
            f"{r['setting']:<22} "
            f"{r['avg_original_tokens']:>12.0f} "
            f"{r['avg_pruned_tokens']:>14.0f} "
            f"{r['avg_token_reduction_pct']:>11.1f}% "
            f"{r['avg_retrieve_rerank_prune_latency_s']:>10.3f}s"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
