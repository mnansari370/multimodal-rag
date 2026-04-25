"""
Dense retriever using sentence-transformers + FAISS.

Encodes all chunks into dense embeddings using a bi-encoder model and
stores them in a FAISS flat index for fast approximate nearest-neighbor
search. BAAI/bge-base-en-v1.5 gives a good quality-to-speed tradeoff.
"""

import os
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
    """
    Bi-encoder retriever backed by a FAISS flat L2 index.

    Usage:
        retriever = DenseRetriever()
        retriever.build(chunks)           # encodes & indexes
        results = retriever.search("...", top_k=20)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        self.dim = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def build(self, chunks: list[dict], show_progress: bool = True) -> None:
        """
        Encode all chunks and build the FAISS index.

        For bge models, prepending 'Represent this sentence for searching
        relevant passages:' to queries (not documents) is recommended.
        We store the raw chunk texts without the prefix.
        """
        self._load_model()
        self.chunks = chunks

        texts = [c["text"] for c in chunks]
        logger.info(f"Encoding {len(texts)} chunks with {self.model_name}...")

        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # cosine similarity via inner product
            convert_to_numpy=True,
        )

        self.dim = embeddings.shape[1]
        # Inner product index — works as cosine similarity because we normalized
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype(np.float32))

        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={self.dim}")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Encode query and search the FAISS index.

        The bge model recommends a task-specific prefix for queries.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        self._load_model()

        # bge recommends this prefix for retrieval queries
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
        """Save the FAISS index and chunk list to disk."""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)

        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        logger.info(f"Dense index saved: {index_path}")

    def load(self, index_path: str, chunks_path: str) -> None:
        """Load a previously saved index and chunk list."""
        self._load_model()
        self.index = faiss.read_index(index_path)

        self.chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line.strip()))

        logger.info(f"Dense index loaded: {self.index.ntotal} vectors, {len(self.chunks)} chunks")
