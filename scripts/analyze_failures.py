"""
Failure analysis script (Section 14 of the project report).

Finds benchmark examples where the retriever misses the gold source and
saves a candidate list for manual inspection and write-up.

The report says to document at least 5 failure cases with one paragraph
each explaining what went wrong. This script finds the raw candidates;
the write-up goes in the report / README.

Common failure types this script flags:
  1. Retriever finds topically related chunks but not the specific fix
  2. Gold source was chunked in a way that separated the relevant part
  3. The query terms are too vague for both BM25 and dense retrieval
  4. The reformulated query over-emphasized the error code vs. the fix

Usage:
    python scripts/analyze_failures.py
    python scripts/analyze_failures.py --top-k 10 --max-failures 20
"""

import sys
import json
import logging
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import MultimodalRAGPipeline, PipelineConfig
from src.evaluation.retrieval_metrics import _normalize_url


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def classify_failure_type(
    query: str,
    gold_urls: set,
    retrieved_urls: list[str],
    all_retrieved: list[dict],
) -> str:
    """
    Guess the most likely failure type by looking at what was retrieved.

    This is a heuristic — use it as a starting point for the manual analysis.
    """
    if not retrieved_urls:
        return "retriever_returned_nothing"

    # Check if any retrieved chunk is from the same page (URL match ignoring #section)
    gold_pages = {u.split("#")[0] for u in gold_urls}
    retrieved_pages = {u.split("#")[0] for u in retrieved_urls}
    if gold_pages & retrieved_pages:
        return "correct_page_wrong_chunk (chunking issue)"

    # Check if BM25 and dense disagreed badly — retrieve with only one of the two
    scores = [r.get("bm25_score", 0) for r in all_retrieved[:5]]
    if all(s == 0 for s in scores):
        return "bm25_found_nothing (sparse query — try reformulation)"

    return "wrong_topic_retrieved (query too vague or missing key terms)"


def main():
    parser = argparse.ArgumentParser(
        description="Find and analyze retrieval failures in the benchmark"
    )
    parser.add_argument(
        "--benchmark",
        default="data/benchmark/benchmark.json",
        help="Path to benchmark JSON",
    )
    parser.add_argument(
        "--output",
        default="results/failure_analysis.json",
        help="Where to save the failure candidates JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of retrieved chunks to check for each query",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=25,
        help="Maximum number of failure cases to save",
    )
    args = parser.parse_args()

    benchmark_path = ROOT / args.benchmark
    output_path = ROOT / args.output

    logger.info("Loading benchmark from %s", benchmark_path)
    with open(benchmark_path, encoding="utf-8") as f:
        benchmark = json.load(f)
    logger.info("Benchmark size: %d examples", len(benchmark))

    # Use the full hybrid + reranker stack so failures are meaningful
    cfg = PipelineConfig(
        retriever_type="hybrid",
        use_reranker=True,
        use_vlp=False,
    )
    pipe = MultimodalRAGPipeline(cfg)
    pipe.load_indexes()

    failures = []
    total = 0
    hit = 0

    for ex in benchmark:
        total += 1
        query = ex.get("log_snippet", "")
        if query:
            query = query.strip() + " " + ex["query"]
        else:
            query = ex["query"]

        gold_urls = {_normalize_url(u) for u in ex.get("gold_source_urls", [])}
        if not gold_urls:
            continue   # nothing to evaluate against

        retrieved = pipe.retriever.search(query, top_k=50)
        reranked = pipe.reranker.rerank(query, retrieved, top_k=args.top_k)

        # Deduplicate by URL, preserving rank order
        seen = set()
        retrieved_urls = []
        for r in reranked:
            u = _normalize_url(r.get("source_url", ""))
            if u not in seen:
                seen.add(u)
                retrieved_urls.append(u)

        found = bool(set(retrieved_urls) & gold_urls)
        if found:
            hit += 1
        else:
            failure_type = classify_failure_type(
                query, gold_urls, retrieved_urls, reranked
            )
            failures.append({
                "id": ex.get("id", f"ex_{total:03d}"),
                "category": ex.get("category"),
                "difficulty": ex.get("difficulty"),
                "query": ex["query"],
                "log_snippet": ex.get("log_snippet", ""),
                "gold_source_urls": ex.get("gold_source_urls", []),
                "top_retrieved_urls": retrieved_urls[:5],
                "failure_type": failure_type,
                "top_reranker_score": reranked[0].get("reranker_score", 0) if reranked else 0,
                "analysis_note": (
                    "Gold source not found in top-{} reranked results. "
                    "Review: query terms, chunking of the gold page, "
                    "and whether reformulation would help."
                ).format(args.top_k),
            })

    recall = hit / max(total, 1)
    logger.info("")
    logger.info("Hit@%d: %.1f%%  (%d / %d)", args.top_k, recall * 100, hit, total)
    logger.info("Failures found: %d", len(failures))

    # Sort by difficulty so harder cases appear first
    difficulty_order = {"hard": 0, "medium": 1, "easy": 2}
    failures.sort(key=lambda x: difficulty_order.get(x.get("difficulty", "easy"), 3))

    top_failures = failures[:args.max_failures]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(top_failures, f, indent=2)

    logger.info("Saved top %d failure cases to %s", len(top_failures), output_path)
    logger.info("")
    logger.info("Next step: read %s and write one paragraph per case for the report.", output_path)


if __name__ == "__main__":
    main()
