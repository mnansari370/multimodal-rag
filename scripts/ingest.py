"""
Full ingestion pipeline — download, clean, and chunk in one command.

This runs the three ingestion stages sequentially:
  1. Download PyTorch documentation
  2. Clean and normalize the raw pages
  3. Chunk using heading-based strategy (with fixed-size as comparison)

Usage:
  python scripts/ingest.py
  python scripts/ingest.py --max-pages 100  # for quick testing
  python scripts/ingest.py --skip-download   # if you already have raw pages
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import download_pytorch_docs, clean_all
from src.chunking import chunk_all


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Full ingestion pipeline")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Limit download to N pages (useful for testing)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if raw pages already exist")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between download requests")
    parser.add_argument("--heading-only", action="store_true",
                        help="Only run heading-based chunking (skip fixed-size, saves memory)")
    args = parser.parse_args()

    if not args.skip_download:
        logger.info("=== Downloading PyTorch documentation ===")
        download_pytorch_docs(args.raw_dir, args.max_pages, args.delay)

    logger.info("=== Cleaning raw pages ===")
    cleaned = clean_all(args.raw_dir, args.processed_dir)
    logger.info(f"Cleaned {len(cleaned)} pages")

    logger.info("=== Chunking (heading-based) ===")
    chunks_heading = chunk_all(
        args.processed_dir,
        output_file=f"{args.processed_dir}/chunks_heading.jsonl",
        strategy="heading",
    )
    logger.info(f"Heading chunks: {len(chunks_heading)}")

    chunks_fixed_count = "skipped"
    if not args.heading_only:
        logger.info("=== Chunking (fixed-size with overlap) ===")
        chunks_fixed = chunk_all(
            args.processed_dir,
            output_file=f"{args.processed_dir}/chunks_fixed.jsonl",
            strategy="fixed",
        )
        logger.info(f"Fixed-size chunks: {len(chunks_fixed)}")
        chunks_fixed_count = len(chunks_fixed)

    print("\n" + "="*50)
    print("Ingestion complete.")
    print(f"  Pages downloaded: {len(list(Path(args.raw_dir).glob('*.json')))}")
    print(f"  Pages cleaned:    {len(cleaned)}")
    print(f"  Chunks (heading): {len(chunks_heading)}")
    print(f"  Chunks (fixed):   {chunks_fixed_count}")
    print(f"\nNext step: python scripts/build_index.py")


if __name__ == "__main__":
    main()
