"""
Retriever ablation — compares BM25-only, dense-only, hybrid, and hybrid+reranker
on the benchmark. Computes Recall@k, Hit@k, MRR, and nDCG@k for each setting.

Usage:
    python scripts/run_retriever_ablation.py
    python scripts/run_retriever_ablation.py --benchmark data/benchmark/benchmark.json
    python scripts/run_retriever_ablation.py --output results/retriever_ablation.json
"""

import sys
import json
import logging
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import MultimodalRAGPipeline, PipelineConfig
from src.evaluation.retrieval_metrics import evaluate_retrieval, print_retrieval_results


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


# (name, retriever_type, use_reranker). VLP is off — isolated retriever comparison.
EXPERIMENTS = [
    ("bm25_only",         "bm25",   False),
    ("dense_only",        "dense",  False),
    ("hybrid",            "hybrid", False),
    ("hybrid_reranker",   "hybrid", True),
]


def main():
    parser = argparse.ArgumentParser(
        description="Run the retriever ablation study"
    )
    parser.add_argument(
        "--benchmark",
        default="data/benchmark/benchmark.json",
        help="Path to the evaluation benchmark JSON",
    )
    parser.add_argument(
        "--output",
        default="results/retriever_ablation.json",
        help="Where to save the results JSON",
    )
    args = parser.parse_args()

    benchmark_path = ROOT / args.benchmark
    output_path = ROOT / args.output

    logger.info("Loading benchmark from %s", benchmark_path)
    with open(benchmark_path, encoding="utf-8") as f:
        benchmark = json.load(f)
    logger.info("Benchmark size: %d examples", len(benchmark))

    all_results = {}

    for name, retriever_type, use_reranker in EXPERIMENTS:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Experiment: %s  (retriever=%s, reranker=%s)", name, retriever_type, use_reranker)
        logger.info("=" * 70)

        cfg = PipelineConfig(
            retriever_type=retriever_type,
            use_reranker=use_reranker,
            use_vlp=False,   # VLP is off — we're isolating the retriever
        )

        pipe = MultimodalRAGPipeline(cfg)
        pipe.load_indexes()

        def make_retrieve_fn(pipe, use_reranker):
            """Closure to capture the right pipe and reranker setting."""
            def retrieve_fn(query):
                candidates = pipe.retriever.search(query, top_k=50)
                if use_reranker:
                    return pipe.reranker.rerank(query, candidates, top_k=20)
                return candidates[:20]
            return retrieve_fn

        retrieve_fn = make_retrieve_fn(pipe, use_reranker)
        metrics = evaluate_retrieval(benchmark, retrieve_fn)

        print_retrieval_results(metrics, label=name)
        all_results[name] = metrics

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info("")
    logger.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()
