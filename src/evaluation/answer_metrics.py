"""
Answer quality evaluation metrics.

Measures how good the generated answers are compared to gold answers
or how well they are grounded in the retrieved context.

Metrics:
  - Token F1         : word-level overlap between predicted and gold answer
  - Semantic similarity : cosine similarity between answer embeddings
  - Citation accuracy   : checks if cited source URLs are correct
  - RAGAS metrics       : faithfulness, answer relevancy, context precision/recall
"""

import re
import logging
from collections import Counter
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


# ─── Token F1 ─────────────────────────────────────────────────────────────────

def _tokenize_answer(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def token_f1(prediction: str, gold: str) -> dict:
    """
    Compute token-level F1 score between prediction and gold answer.
    Similar to SQuAD evaluation.
    """
    pred_tokens = _tokenize_answer(prediction)
    gold_tokens = _tokenize_answer(gold)

    if not pred_tokens or not gold_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common = sum((pred_counter & gold_counter).values())

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4)}


# ─── Semantic similarity ──────────────────────────────────────────────────────

class SemanticSimilarity:
    """
    Embeds answers with a sentence-transformer and measures cosine similarity.
    More robust than token F1 for paraphrased answers.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def score(self, prediction: str, gold: str) -> float:
        self._load()
        embs = self._model.encode([prediction, gold], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))

    def batch_score(self, predictions: list[str], golds: list[str]) -> list[float]:
        self._load()
        all_texts = predictions + golds
        embs = self._model.encode(all_texts, normalize_embeddings=True)
        pred_embs = embs[:len(predictions)]
        gold_embs = embs[len(predictions):]
        return [float(np.dot(p, g)) for p, g in zip(pred_embs, gold_embs)]


# ─── Citation accuracy ────────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    """Strip domain prefix for URL comparison — same logic as retrieval_metrics."""
    for prefix in [
        "https://docs.pytorch.org",
        "http://docs.pytorch.org",
        "https://pytorch.org",
        "http://pytorch.org",
    ]:
        if url.startswith(prefix):
            return url[len(prefix):].rstrip("/")
    return url.rstrip("/")


def citation_accuracy(answer: str, gold_urls: list[str], source_chunks: list[dict]) -> float:
    """
    Check what fraction of [Source N] citations in the answer point to
    correct documentation URLs.

    Args:
        answer: Generated answer text containing [Source N] markers.
        gold_urls: List of correct source URLs for this query.
        source_chunks: Chunks that were passed as context (in order, 1-indexed).

    Returns:
        Fraction of cited sources that are in gold_urls. 1.0 = all correct.
    """
    cited_indices = [int(m) - 1 for m in re.findall(r"\[Source (\d+)\]", answer)]

    if not cited_indices:
        return 0.0

    gold_url_set = {_normalize_url(u) for u in gold_urls}
    correct = 0
    for idx in cited_indices:
        if 0 <= idx < len(source_chunks):
            chunk_url = _normalize_url(source_chunks[idx].get("source_url", ""))
            if chunk_url in gold_url_set:
                correct += 1

    return round(correct / len(cited_indices), 4)


# ─── RAGAS evaluation ─────────────────────────────────────────────────────────

def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """
    Run RAGAS evaluation on a batch of (question, answer, context, ground_truth) tuples.

    RAGAS measures:
      - faithfulness    : are the claims in the answer supported by the context?
      - answer_relevancy: is the answer relevant to the question?
      - context_precision: are the retrieved chunks relevant?
      - context_recall   : does the context cover the gold answer?

    Requires OPENAI_API_KEY or ANTHROPIC_API_KEY to be set (RAGAS uses LLM judge).

    Returns:
        Dict of metric name → score (averaged over the batch).
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError:
        logger.warning("ragas not installed. Run: pip install ragas datasets")
        return {}

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        return {k: round(float(v), 4) for k, v in result.items()}
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {}


# ─── Full answer evaluation ───────────────────────────────────────────────────

def evaluate_answers(
    benchmark: list[dict],
    predictions: list[str],
    source_chunks_per_query: list[list[dict]],
    run_ragas: bool = True,
) -> dict:
    """
    Evaluate a list of predicted answers against the benchmark.

    Each benchmark example needs: 'query', 'gold_answer', 'gold_source_urls'.

    Args:
        benchmark: List of evaluation examples.
        predictions: Generated answers (same order as benchmark).
        source_chunks_per_query: Context chunks used for each answer.
        run_ragas: Whether to run RAGAS (requires API key).

    Returns:
        Aggregated metrics dict.
    """
    f1_scores = []
    citation_scores = []

    for example, pred, chunks in zip(benchmark, predictions, source_chunks_per_query):
        gold = example.get("gold_answer", "")
        gold_urls = example.get("gold_source_urls", [])

        f1_result = token_f1(pred, gold)
        f1_scores.append(f1_result["f1"])

        cit = citation_accuracy(pred, gold_urls, chunks)
        citation_scores.append(cit)

    results = {
        "token_f1": round(sum(f1_scores) / max(len(f1_scores), 1), 4),
        "citation_accuracy": round(sum(citation_scores) / max(len(citation_scores), 1), 4),
    }

    if run_ragas:
        ragas_results = run_ragas_evaluation(
            questions=[e["query"] for e in benchmark],
            answers=predictions,
            contexts=[[c["text"] for c in chunks] for chunks in source_chunks_per_query],
            ground_truths=[e.get("gold_answer", "") for e in benchmark],
        )
        results.update(ragas_results)

    return results
