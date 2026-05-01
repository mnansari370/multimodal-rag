"""
Context selector — the core innovation of this project.

After reranking we have a ranked list of evidence chunks. Sending all of
them to the generator wastes tokens, increases latency, and can hurt answer
quality because the model gets distracted by irrelevant chunks.

Three pruning strategies, selectable at runtime for the ablation study:

  threshold    — keep every chunk whose reranker score exceeds a tuned
                 threshold. Simple baseline, easy to ablate.

  diversity    — top-k selection that penalizes near-duplicates using
                 character n-gram Jaccard similarity between chunks.
                 Balances relevance and coverage.

  coverage     — MMR-style selection that scores each candidate by how
                 much new information it adds relative to already-selected
                 chunks. Best for citation quality.
"""

import logging
from typing import Literal


logger = logging.getLogger(__name__)


def prune_by_threshold(
    chunks: list[dict],
    threshold: float = 0.0,
    min_chunks: int = 1,
    max_chunks: int = 10,
) -> list[dict]:
    """
    Keep chunks whose reranker_score exceeds threshold.

    If fewer than min_chunks pass, return the top min_chunks anyway
    so the generator always has something to work with.
    """
    above = [c for c in chunks if c.get("reranker_score", 0.0) >= threshold]

    if len(above) < min_chunks:
        above = chunks[:min_chunks]

    return above[:max_chunks]


def prune_top_k_diverse(
    chunks: list[dict],
    top_k: int = 5,
    similarity_penalty_threshold: float = 0.85,
) -> list[dict]:
    """
    Greedy diversity-aware selection using character n-gram Jaccard similarity.

    Starts with the highest-scored chunk, then skips candidates that are too
    similar to any already-selected chunk. No embedding needed at runtime.
    """
    if not chunks:
        return []

    def jaccard(a: str, b: str, n: int = 4) -> float:
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


def prune_by_coverage(
    chunks: list[dict],
    query: str,
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance (MMR) style selection.

    Scores each candidate by:
        MMR(d) = λ * relevance(d) - (1-λ) * max_sim(d, selected)

    where relevance = normalized reranker_score and max_sim is n-gram
    Jaccard similarity to the already-selected set.

    λ=1.0 → pure relevance (same as top-k); λ=0.0 → pure diversity.
    """
    if not chunks:
        return []

    def ngram_jaccard(a: str, b: str, n: int = 4) -> float:
        ngrams_a = {a[i:i+n] for i in range(len(a) - n + 1)}
        ngrams_b = {b[i:i+n] for i in range(len(b) - n + 1)}
        if not ngrams_a or not ngrams_b:
            return 0.0
        return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)

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

    Returns (selected_chunks, stats) where stats contains token counts
    and reduction percentage for efficiency tracking.
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
        "Context pruning (%s): %d → %d chunks, %s%% token reduction",
        strategy, len(chunks), len(selected), stats["token_reduction_pct"],
    )

    return selected, stats
