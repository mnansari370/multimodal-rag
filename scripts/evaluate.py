"""
Run the full evaluation suite on the benchmark.

Produces:
  - Retrieval metrics (Recall@k, MRR, nDCG@k)
  - Answer quality metrics (Token F1, citation accuracy)
  - RAGAS metrics (faithfulness, answer relevancy, context precision/recall)
  - Efficiency metrics (token count, latency)

Results are saved to results/evaluation_<timestamp>.json

Usage:
  python scripts/evaluate.py \
    --benchmark data/benchmark/benchmark.json \
    --config configs/pipeline.yaml \
    --output results/evaluation.json
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import MultimodalRAGPipeline, PipelineConfig
from src.evaluation import (
    evaluate_retrieval,
    evaluate_answers,
    compute_efficiency_stats,
    print_efficiency_table,
    print_retrieval_results,
)
from src.evaluation.efficiency_metrics import LatencyProfile


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Full evaluation suite")
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.json")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-ragas", action="store_true", help="Skip RAGAS (no API key)")
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    with open(args.benchmark) as f:
        benchmark = json.load(f)

    if args.max_examples:
        benchmark = benchmark[:args.max_examples]

    logger.info(f"Running evaluation on {len(benchmark)} examples")

    # Build pipeline from config
    cfg = PipelineConfig()
    if Path(args.config).exists():
        import yaml
        with open(args.config) as f:
            data = yaml.safe_load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    pipeline = MultimodalRAGPipeline(cfg)
    pipeline.load_indexes()

    # ── Retrieval evaluation ──────────────────────────────────────────────────
    logger.info("Running retrieval evaluation...")

    def retrieve_fn(query):
        return pipeline.retriever.search(query, top_k=20)

    retrieval_metrics = evaluate_retrieval(benchmark, retrieve_fn)
    print_retrieval_results(retrieval_metrics)

    # ── End-to-end generation evaluation ─────────────────────────────────────
    logger.info("Running generation evaluation...")

    predictions = []
    all_chunks = []
    pruning_stats_list = []
    latency_profiles = []

    for i, example in enumerate(benchmark):
        logger.info(f"  Example {i+1}/{len(benchmark)}: {example['query'][:50]}")

        result = pipeline.run(
            question=example["query"],
            log_snippet=example.get("log_snippet", ""),
        )

        predictions.append(result.answer)
        all_chunks.append(result.selected_chunks)
        pruning_stats_list.append(result.pruning_stats)
        if result.latency:
            latency_profiles.append(result.latency)

    answer_metrics = evaluate_answers(
        benchmark, predictions, all_chunks, run_ragas=not args.no_ragas
    )

    efficiency_metrics = compute_efficiency_stats(pruning_stats_list, latency_profiles)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n=== RETRIEVAL METRICS ===")
    for k, v in sorted(retrieval_metrics.items()):
        print(f"  {k:<20} {v:.4f}")

    print("\n=== ANSWER QUALITY METRICS ===")
    for k, v in sorted(answer_metrics.items()):
        print(f"  {k:<25} {v}")

    print("\n=== EFFICIENCY METRICS ===")
    for k, v in sorted(efficiency_metrics.items()):
        print(f"  {k:<35} {v}")

    # ── Save results ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"results/evaluation_{timestamp}.json"

    all_results = {
        "timestamp": timestamp,
        "benchmark_size": len(benchmark),
        "config": {
            "retriever_type": cfg.retriever_type,
            "pruning_strategy": cfg.pruning_strategy,
            "context_top_k": cfg.context_top_k,
            "generator_model": cfg.generator_model,
        },
        "retrieval_metrics": retrieval_metrics,
        "answer_metrics": answer_metrics,
        "efficiency_metrics": efficiency_metrics,
    }

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
