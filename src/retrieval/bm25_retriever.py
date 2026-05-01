"""
BM25 sparse retriever.

BM25 is a strong baseline for keyword-heavy queries like error messages
and function names. It needs no GPU and builds in seconds.
"""

import pickle
import logging
from pathlib import Path

from rank_bm25 import BM25Okapi


logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace + punctuation."""
    import re
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class BM25Retriever:
    """Sparse retriever backed by rank_bm25.BM25Okapi."""

    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def build(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built over {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Return top_k chunks ranked by BM25 score, each with a 'bm25_score' field."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        scored = sorted(
            zip(scores, self.chunks),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        results = []
        for score, chunk in scored:
            result = dict(chunk)
            result["bm25_score"] = float(score)
            results.append(result)

        return results

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        logger.info(f"BM25 index saved to {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        logger.info(f"BM25 index loaded from {path} ({len(self.chunks)} chunks)")
