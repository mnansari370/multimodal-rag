"""
Run the VLP (Vision-Language Parser) over all benchmark screenshots and
cache the structured outputs to disk.

Caching the outputs means downstream evaluations can re-run without
re-calling the vision API every time.

Output format (one entry per benchmark example):
  {
    "ex_001": {
      "error_message": "...",
      "visual_category": "stack_trace",
      "software_components": ["PyTorch", "CUDA"],
      "keywords": ["OOM", "batch_size", ...],
      "raw_description": "..."
    },
    "ex_002": null,   <- example had no image
    ...
  }

Usage:
    python scripts/run_vlp_batch.py
    python scripts/run_vlp_batch.py --backend anthropic
    python scripts/run_vlp_batch.py --backend internvl2 --model OpenGVLab/InternVL2-2B
"""

import sys
import json
import logging
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.multimodal import VisionLanguageParser


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run VLP in batch over all benchmark screenshots"
    )
    parser.add_argument(
        "--benchmark",
        default="data/benchmark/benchmark.json",
        help="Path to benchmark JSON",
    )
    parser.add_argument(
        "--output",
        default="results/vlp_outputs.json",
        help="Where to save VLP outputs",
    )
    parser.add_argument(
        "--backend",
        default="anthropic",
        choices=["anthropic", "openai", "internvl2"],
        help="Which VLM backend to use",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the default model for the chosen backend",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip examples already present in the output file",
    )
    args = parser.parse_args()

    benchmark_path = ROOT / args.benchmark
    output_path = ROOT / args.output

    logger.info("Loading benchmark from %s", benchmark_path)
    with open(benchmark_path, encoding="utf-8") as f:
        benchmark = json.load(f)

    screenshot_count = sum(
        1 for ex in benchmark
        if ex.get("image_path") and Path(ex["image_path"]).exists()
    )
    logger.info(
        "Benchmark: %d examples, %d with existing screenshots",
        len(benchmark), screenshot_count,
    )

    # Load any existing outputs if resuming
    existing = {}
    if args.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        logger.info("Resuming: %d outputs already cached", len(existing))

    # Initialize VLP — for anthropic/openai, no GPU needed
    logger.info("Initializing VLP backend=%s", args.backend)
    vlp = VisionLanguageParser(backend=args.backend, model_name=args.model)

    outputs = dict(existing)
    processed = 0
    skipped = 0
    failed = 0

    for ex in benchmark:
        ex_id = ex["id"]

        if ex_id in outputs:
            skipped += 1
            continue

        img_path = ex.get("image_path")
        if not img_path or not Path(img_path).exists():
            outputs[ex_id] = None
            continue

        logger.info("Processing %s: %s", ex_id, img_path)
        try:
            result = vlp.parse(image_path=img_path)
            outputs[ex_id] = result.to_dict()
            processed += 1
            logger.info(
                "  → category=%s  error=%s",
                result.visual_category,
                result.error_message[:60],
            )
        except Exception as e:
            logger.error("  Failed on %s: %s", ex_id, e)
            outputs[ex_id] = {"error": str(e)}
            failed += 1

        # Save after each example so a partial run isn't lost
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2)

    logger.info("")
    logger.info(
        "Done — processed=%d  skipped=%d  failed=%d",
        processed, skipped, failed,
    )
    logger.info("VLP outputs saved to %s", output_path)


if __name__ == "__main__":
    main()
