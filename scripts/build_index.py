"""
Build retrieval indexes from chunked documentation.

Run this after corpus_build.sh (or the downloader/cleaner) to create
the FAISS and BM25 indexes needed by the pipeline.

Usage:
  python scripts/build_index.py --chunks data/processed/chunks_heading.jsonl
"""

import sys
import argparse
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import HybridRetriever


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indexes")
    parser.add_argument("--chunks", default="data/processed/chunks_heading.jsonl",
                        help="Path to chunks JSONL file")
    parser.add_argument("--bm25-out", default="data/embeddings/bm25.pkl")
    parser.add_argument("--faiss-out", default="data/embeddings/dense.faiss")
    parser.add_argument("--chunks-out", default="data/embeddings/chunks.jsonl")
    args = parser.parse_args()

    logger.info(f"Loading chunks from {args.chunks}")
    chunks = []
    with open(args.chunks) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    logger.info(f"Loaded {len(chunks)} chunks")

    retriever = HybridRetriever()
    retriever.build(chunks)
    retriever.save(
        bm25_path=args.bm25_out,
        faiss_path=args.faiss_out,
        chunks_path=args.chunks_out,
    )

    logger.info("Index building complete.")
    print(f"\nIndexes saved to:")
    print(f"  BM25:   {args.bm25_out}")
    print(f"  FAISS:  {args.faiss_out}")
    print(f"  Chunks: {args.chunks_out}")


if __name__ == "__main__":
    main()
