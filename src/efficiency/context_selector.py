"""
Context selector — the core innovation of this project.

After reranking we have a ranked list of evidence chunks. Sending ALL of
them to the generator wastes tokens, increases latency, and can actually
hurt answer quality because the model gets distracted by irrelevant chunks.

This module implements three pruning strategies that you can swap in
and compare in the ablation study (Section 6 of the project report):

  1. score_threshold  — keep every chunk whose reranker score exceeds a
                        tuned threshold. Simple baseline.

  2. top_k_diversity  — keep the top-k chunks but penalize near-duplicates
                        using cosine similarity between already-selected
                        chunks. Balances relevance and diversity.

  3. coverage_pruning — MMR-style: score each candidate by how much NEW
                        information it adds relative to already-selected
                        chunks. Strongest for citation quality.

The ablation table to produce (Section 6.2):
  Setting | Prompt tokens | Latency | Faithfulness | Answer score
"""

import logging
import numpy as np
from typing import Literal


logger = logging.getLogger(__name__)


# ─── Strategy 1: Score threshold ─────────────────────────────────────────────

def prune_by_threshold(
    chunks: list[dict],
    threshold: float = 0.0,
    min_chunks: int = 1,
    max_chunks: int = 10,
) -> list[dict]:
    """
    Keep chunks whose reranker_score exceeds `threshold`.

    If fewer than min_chunks pass the threshold we return the top min_chunks
    anyway (safety fallback). Never returns more than max_chunks.

    Args:
        chunks: Reranked chunks in score-descending order.
        threshold: Minimum reranker score to keep (tune on validation set).
        min_chunks: Always return at least this many.
        max_chunks: Hard cap on context size.
    """
    above = [c for c in chunks if c.get("reranker_score", 0.0) >= threshold]

    if len(above) < min_chunks:
        above = chunks[:min_chunks]

    return above[:max_chunks]


# ─── Strategy 2: Top-k + diversity ───────────────────────────────────────────

def prune_top_k_diverse(
    chunks: list[dict],
    top_k: int = 5,
    similarity_penalty_threshold: float = 0.85,
) -> list[dict]:
    """
    Greedy diversity-aware selection.

    Start with the highest-scored chunk. For each subsequent candidate,
    skip it if it is too similar (above similarity_penalty_threshold) to
    any already-selected chunk. Continue until we have top_k chunks.

    Uses character-level n-gram overlap as a cheap similarity proxy
    (no embedding needed at runtime).

    Args:
        chunks: Reranked chunks in score-descending order.
        top_k: Target number of chunks to keep.
        similarity_penalty_threshold: Jaccard similarity above which a
            chunk is considered a near-duplicate of a selected chunk.
    """
    if not chunks:
        return []

    def jaccard(a: str, b: str, n: int = 4) -> float:
        """Character-level n-gram Jaccard similarity."""
        ngrams_a = {a[i:i+n] for i in range(len(a) - n + 1)}
        ngrams_b = {b[i:i+n] for i in range(len(b) - n + 1)}
        if not ngrams_a or not ngrams_b:
            return 0.0
        return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)

    selected = [chunks[0]]
    selected_texts = [chunks[0]["text"]]

    for candidate in chunks[1:]:
        if len(selected) >= top_k:
            break
        candidate_text = candidate["text"]
        max_sim = max(jaccard(candidate_text, sel_text) for sel_text in selected_texts)
        if max_sim < similarity_penalty_threshold:
            selected.append(candidate)
            selected_texts.append(candidate_text)

    return selected


# ─── Strategy 3: Coverage pruning (MMR-style) ────────────────────────────────

def prune_by_coverage(
    chunks: list[dict],
    query: str,
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance (MMR) style coverage pruning.

    Scores each candidate by:
      MMR(d) = λ * relevance(d) - (1-λ) * max_sim(d, selected)

    relevance = normalized reranker_score
    max_sim   = maximum n-gram Jaccard similarity to any already-selected chunk

    Lambda=1.0 → pure relevance (same as top-k)
    Lambda=0.0 → pure diversity

    Args:
        chunks: Reranked chunks with 'reranker_score'.
        query: Original query (not used in this implementation but
               kept as parameter for future embedding-based scoring).
        top_k: Number of chunks to select.
        lambda_param: Trade-off between relevance and diversity.
    """
    if not chunks:
        return []

    def ngram_jaccard(a: str, b: str, n: int = 4) -> float:
        ngrams_a = {a[i:i+n] for i in range(len(a) - n + 1)}
        ngrams_b = {b[i:i+n] for i in range(len(b) - n + 1)}
        if not ngrams_a or not ngrams_b:
            return 0.0
        return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)

    # Normalize scores to [0, 1]
    raw_scores = [c.get("reranker_score", 0.0) for c in chunks]
    min_s, max_s = min(raw_scores), max(raw_scores)
    score_range = max(max_s - min_s, 1e-9)
    rel_scores = [(s - min_s) / score_range for s in raw_scores]

    remaining = list(zip(rel_scores, chunks))
    selected = []
    selected_texts = []

    while remaining and len(selected) < top_k:
        best_score = -float("inf")
        best_idx = 0

        for i, (rel, candidate) in enumerate(remaining):
            if not selected_texts:
                mmr = rel
            else:
                max_sim = max(ngram_jaccard(candidate["text"], t) for t in selected_texts)
                mmr = lambda_param * rel - (1.0 - lambda_param) * max_sim

            if mmr > best_score:
                best_score = mmr
                best_idx = i

        _, chosen = remaining.pop(best_idx)
        selected.append(chosen)
        selected_texts.append(chosen["text"])

    return selected


# ─── Unified interface ────────────────────────────────────────────────────────

def select_context(
    chunks: list[dict],
    strategy: Literal["threshold", "diversity", "coverage"] = "threshold",
    query: str = "",
    threshold: float = 0.0,
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> tuple[list[dict], dict]:
    """
    Select the final context chunks using the chosen pruning strategy.

    Args:
        chunks: Reranked chunks (score-descending).
        strategy: Which pruning method to use.
        query: Needed for coverage strategy.
        threshold: For 'threshold' strategy.
        top_k: For 'diversity' and 'coverage' strategies.
        lambda_param: For 'coverage' strategy.

    Returns:
        (selected_chunks, stats) where stats reports token counts.
    """
    original_token_count = sum(c.get("approx_tokens", len(c["text"]) // 4) for c in chunks)

    if strategy == "threshold":
        selected = prune_by_threshold(chunks, threshold=threshold, max_chunks=top_k)
    elif strategy == "diversity":
        selected = prune_top_k_diverse(chunks, top_k=top_k)
    elif strategy == "coverage":
        selected = prune_by_coverage(chunks, query=query, top_k=top_k, lambda_param=lambda_param)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    pruned_token_count = sum(c.get("approx_tokens", len(c["text"]) // 4) for c in selected)

    stats = {
        "strategy": strategy,
        "input_chunks": len(chunks),
        "selected_chunks": len(selected),
        "original_tokens": original_token_count,
        "pruned_tokens": pruned_token_count,
        "token_reduction_pct": round(
            100 * (1 - pruned_token_count / max(original_token_count, 1)), 1
        ),
    }

    logger.debug(
        f"Context pruning ({strategy}): {len(chunks)} → {len(selected)} chunks, "
        f"{stats['token_reduction_pct']}% token reduction"
    )

    return selected, stats
