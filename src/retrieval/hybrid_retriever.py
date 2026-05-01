"""
Hybrid retriever using Reciprocal Rank Fusion (RRF).

Combines BM25 (keyword) and dense (semantic) results. RRF is simple
and consistently outperforms linear score interpolation because it
doesn't require tuning score weights — rank position alone drives the
fusion.

RRF formula:  score(d) = Σ  1 / (k + rank(d)),   k=60 standard constant.
"""

import logging
from collections import defaultdict

from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever


logger = logging.getLogger(__name__)

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    score_keys: list[str],
    id_key: str = "chunk_id",
    k: int = RRF_K,
) -> list[dict]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        rankings: Ranked result lists from different retrievers.
        score_keys: Original score field names to preserve in output.
        id_key: Field used to match the same chunk across lists.
        k: Smoothing constant (default 60).
    """
    rrf_scores = defaultdict(float)
    chunk_store = {}

    for ranked_list in rankings:
        for rank, chunk in enumerate(ranked_list, start=1):
            cid = chunk[id_key]
            rrf_scores[cid] += 1.0 / (k + rank)
            if cid not in chunk_store:
                chunk_store[cid] = chunk
            else:
                chunk_store[cid].update(
                    {sk: chunk[sk] for sk in score_keys if sk in chunk}
                )

    merged = []
    for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        result = dict(chunk_store[cid])
        result["rrf_score"] = score
        merged.append(result)

    return merged


class HybridRetriever:
    """
    Combines BM25 and dense retrieval via RRF.

    BM25 catches exact error strings and function names; dense catches
    semantic matches when the user's wording differs from the docs.
    The hybrid almost always beats either retriever alone.
    """

    def __init__(
        self,
        bm25: BM25Retriever = None,
        dense: DenseRetriever = None,
    ):
        self.bm25 = bm25 or BM25Retriever()
        self.dense = dense or DenseRetriever()

    def build(self, chunks: list[dict], show_progress: bool = True) -> None:
        """Build both BM25 and dense indexes."""
        logger.info("Building BM25 index...")
        self.bm25.build(chunks)
        logger.info("Building dense index...")
        self.dense.build(chunks, show_progress=show_progress)

    def search(
        self,
        query: str,
        top_k: int = 20,
        bm25_candidates: int = 50,
        dense_candidates: int = 50,
    ) -> list[dict]:
        """Retrieve top_k chunks using RRF over BM25 + dense candidates."""
        bm25_results = self.bm25.search(query, top_k=bm25_candidates)
        dense_results = self.dense.search(query, top_k=dense_candidates)

        fused = reciprocal_rank_fusion(
            rankings=[bm25_results, dense_results],
            score_keys=["bm25_score", "dense_score"],
        )

        return fused[:top_k]

    def save(
        self,
        bm25_path: str = "data/embeddings/bm25.pkl",
        faiss_path: str = "data/embeddings/dense.faiss",
        chunks_path: str = "data/embeddings/chunks.jsonl",
    ) -> None:
        self.bm25.save(bm25_path)
        self.dense.save(faiss_path, chunks_path)

    def load(
        self,
        bm25_path: str = "data/embeddings/bm25.pkl",
        faiss_path: str = "data/embeddings/dense.faiss",
        chunks_path: str = "data/embeddings/chunks.jsonl",
    ) -> None:
        self.bm25.load(bm25_path)
        self.dense.load(faiss_path, chunks_path)
