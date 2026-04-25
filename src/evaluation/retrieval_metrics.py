"""
Retrieval evaluation metrics.

Measures how well the system finds the right documentation chunks
given a benchmark with known gold source URLs or chunk IDs.

Metrics implemented:
  - Recall@k     : Was the gold chunk in the top-k results?
  - Hit@k        : Binary version of Recall@k (1 or 0 per query)
  - MRR           : Mean Reciprocal Rank — where does the gold chunk appear?
  - nDCG@k        : Normalized Discounted Cumulative Gain
"""

import math
import logging
from collections import defaultdict


logger = logging.getLogger(__name__)


def recall_at_k(
    retrieved_ids: list[str],
    gold_ids: set[str],
    k: int,
) -> float:
    """
    Proportion of gold sources found in the top-k results.

    For most queries there is one gold source, so this is 0 or 1.
    For multi-source queries it can be fractional.
    """
    top_k = set(retrieved_ids[:k])
    if not gold_ids:
        return 0.0
    return len(top_k & gold_ids) / len(gold_ids)


def hit_at_k(
    retrieved_ids: list[str],
    gold_ids: set[str],
    k: int,
) -> int:
    """Binary: 1 if any gold source appears in top-k, else 0."""
    return int(bool(set(retrieved_ids[:k]) & gold_ids))


def reciprocal_rank(
    retrieved_ids: list[str],
    gold_ids: set[str],
) -> float:
    """
    1 / rank of the first gold result.

    Returns 0.0 if no gold result appears in the retrieved list.
    """
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    gold_ids: set[str],
    k: int,
) -> float:
    """
    nDCG@k. Penalizes correct answers that appear lower in the list.

    Uses binary relevance: 1 if in gold_ids, 0 otherwise.
    """
    def dcg(ids, gold, k):
        score = 0.0
        for i, rid in enumerate(ids[:k], start=1):
            if rid in gold:
                score += 1.0 / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved_ids, gold_ids, k)
    ideal_hits = min(len(gold_ids), k)
    ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def _normalize_url(url: str) -> str:
    """
    Normalize PyTorch doc URLs so that docs.pytorch.org and pytorch.org/docs
    both map to the same canonical form.

    The corpus uses docs.pytorch.org (the new CDN domain), while human-curated
    benchmarks often reference the older pytorch.org/docs form. Both point to
    the same content, so we strip the domain and keep only the path.
    """
    for prefix in [
        "https://docs.pytorch.org",
        "http://docs.pytorch.org",
        "https://pytorch.org",
        "http://pytorch.org",
    ]:
        if url.startswith(prefix):
            return url[len(prefix):].rstrip("/")
    return url.rstrip("/")


def evaluate_retrieval(
    benchmark: list[dict],
    retrieve_fn,
    k_values: list[int] = [5, 10, 20],
) -> dict:
    """
    Run retrieval evaluation over a benchmark.

    Args:
        benchmark: List of dicts with 'query' and 'gold_chunk_ids' (list of str).
        retrieve_fn: Callable(query: str) → list of chunk dicts with 'chunk_id'.
        k_values: Which k values to compute metrics at.

    Returns:
        Dict of metric → value, averaged over all queries.
    """
    metrics = defaultdict(list)

    for example in benchmark:
        query = example["query"]
        gold_ids = set(example.get("gold_chunk_ids", []))
        gold_urls = set(_normalize_url(u) for u in example.get("gold_source_urls", []))

        results = retrieve_fn(query)
        retrieved_ids = [r.get("chunk_id", "") for r in results]

        # Deduplicate URLs while preserving rank order — multiple chunks from the
        # same page should count as one hit (the page is either relevant or not)
        seen_urls = set()
        deduped_urls = []
        for r in results:
            norm_url = _normalize_url(r.get("source_url", ""))
            if norm_url not in seen_urls:
                seen_urls.add(norm_url)
                deduped_urls.append(norm_url)

        # Evaluate against chunk IDs if available, else deduplicated URLs
        if gold_ids:
            eval_ids = retrieved_ids
            eval_gold = gold_ids
        else:
            eval_ids = deduped_urls
            eval_gold = gold_urls

        metrics["mrr"].append(reciprocal_rank(eval_ids, eval_gold))

        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(eval_ids, eval_gold, k))
            metrics[f"hit@{k}"].append(hit_at_k(eval_ids, eval_gold, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(eval_ids, eval_gold, k))

    aggregated = {k: round(sum(v) / len(v), 4) for k, v in metrics.items() if v}
    return aggregated


def print_retrieval_results(results: dict, label: str = ""):
    header = f"=== Retrieval Metrics {f'({label})' if label else ''} ==="
    print(header)
    for metric, value in sorted(results.items()):
        print(f"  {metric:<15} {value:.4f}")
