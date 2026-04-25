"""
Cross-encoder reranker.

The retriever pulls ~50 candidates quickly using approximate methods.
This reranker then scores each (query, chunk) pair with a cross-encoder
model that sees both together — much more accurate than the bi-encoder
used in retrieval, but too slow to run on the entire corpus.

This is the standard two-stage design: fast retrieval + accurate reranking.
"""

import logging
from sentence_transformers import CrossEncoder


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Cross-encoder reranker wrapping sentence-transformers CrossEncoder.

    Usage:
        reranker = Reranker()
        candidates = hybrid_retriever.search(query, top_k=50)
        reranked = reranker.rerank(query, candidates, top_k=10)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """
        Score each candidate with the cross-encoder and return top_k.

        Args:
            query: The user's (possibly reformulated) query.
            candidates: List of chunk dicts from the retriever.
            top_k: How many to return after reranking.

        Returns:
            Top_k chunks sorted by descending reranker score.
            Each chunk gets a 'reranker_score' field added.
        """
        if not candidates:
            return []

        self._load_model()

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        results = []
        for score, chunk in scored:
            result = dict(chunk)
            result["reranker_score"] = float(score)
            results.append(result)

        return results
