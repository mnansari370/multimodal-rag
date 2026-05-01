"""
Dense retriever using sentence-transformers + FAISS.

Encodes all chunks with a bi-encoder and stores them in a FAISS flat index
for approximate nearest-neighbor search. BAAI/bge-base-en-v1.5 gives a
good quality-to-speed tradeoff and is recommended by the MTEB leaderboard
for retrieval tasks.
"""

import json
import logging
import numpy as np
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 64


class DenseRetriever:
    """Bi-encoder retriever backed by a FAISS flat IP index."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        self.dim = None

    def _load_model(self):
        if self.model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self.model = SentenceTransformer(self.model_name)

    def build(self, chunks: list[dict], show_progress: bool = True) -> None:
        """Encode all chunks and build the FAISS index."""
        self._load_model()
        self.chunks = chunks

        texts = [c["text"] for c in chunks]
        logger.info("Encoding %d chunks with %s...", len(texts), self.model_name)

        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # enables cosine similarity via inner product
            convert_to_numpy=True,
        )

        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype(np.float32))

        logger.info("FAISS index built: %d vectors, dim=%d", self.index.ntotal, self.dim)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Encode query and search the FAISS index."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        self._load_model()

        # bge recommends a task-specific prefix for retrieval queries (not documents)
        prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
        q_emb = self.model.encode(
            [prefixed_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            result = dict(self.chunks[idx])
            result["dense_score"] = float(score)
            results.append(result)

        return results

    def save(self, index_path: str, chunks_path: str) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)

        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        logger.info("Dense index saved: %s", index_path)

    def load(self, index_path: str, chunks_path: str) -> None:
        self._load_model()
        self.index = faiss.read_index(index_path)

        self.chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line.strip()))

        logger.info("Dense index loaded: %d vectors, %d chunks", self.index.ntotal, len(self.chunks))
