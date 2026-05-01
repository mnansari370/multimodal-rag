"""
Compare heading-based vs fixed-size chunking on the benchmark.

Builds a HybridRetriever for each chunking strategy and evaluates
Recall@k, MRR, and nDCG@k to decide which chunking to use for the
final system.

Usage:
    python scripts/run_chunking_comparison.py
"""

import sys
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval import HybridRetriever
from src.evaluation.retrieval_metrics import evaluate_retrieval, print_retrieval_results


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


EXPERIMENTS = [
    ("heading", ROOT / "data/processed/chunks_heading.jsonl"),
    ("fixed",   ROOT / "data/processed/chunks_fixed.jsonl"),
]


def main():
    with open(ROOT / "data/benchmark/benchmark.json", encoding="utf-8") as f:
        benchmark = json.load(f)

    all_results = {}

    for name, chunks_path in EXPERIMENTS:
        logger.info("=" * 80)
        logger.info("Strategy: %s  (%s)", name, chunks_path)
        logger.info("=" * 80)

        chunks = []
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        retriever = HybridRetriever()
        retriever.build(chunks, show_progress=True)

        def retrieve_fn(query, _r=retriever):
            return _r.search(query, top_k=20)

        metrics = evaluate_retrieval(benchmark, retrieve_fn)
        print_retrieval_results(metrics, label=name)
        all_results[name] = metrics

    out = ROOT / "results/chunking_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info("Saved: %s", out)


if __name__ == "__main__":
    main()
