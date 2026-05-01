"""
Interactive command-line interface for the RAG pipeline.

Run a single query through the full pipeline and print the result.

Usage:
  python scripts/run_pipeline.py \
    --question "Why is my training crashing?" \
    --log "RuntimeError: CUDA out of memory" \
    --image path/to/screenshot.png
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MultimodalRAGPipeline, PipelineConfig


logging.basicConfig(level=logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Run the Multimodal RAG pipeline")
    parser.add_argument("--question", required=True, help="Your technical question")
    parser.add_argument("--image", default=None, help="Path to a screenshot image")
    parser.add_argument("--log", default="", help="Log or config snippet (as text)")
    parser.add_argument("--retriever", default="hybrid", choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--pruning", default="threshold", choices=["threshold", "diversity", "coverage"])
    parser.add_argument("--top-k", type=int, default=5, help="Number of context chunks to use")
    parser.add_argument("--backend", default="openai", choices=["local", "openai", "anthropic"])
    parser.add_argument("--model", default=None, help="Override the generator model name")
    parser.add_argument("--no-vlp", action="store_true", help="Disable vision-language parser")
    args = parser.parse_args()

    cfg = PipelineConfig(
        retriever_type=args.retriever,
        pruning_strategy=args.pruning,
        context_top_k=args.top_k,
        generator_backend=args.backend,
        generator_model=args.model or ("gpt-4o-mini" if args.backend == "openai" else None),
        use_vlp=not args.no_vlp and args.image is not None,
    )

    pipeline = MultimodalRAGPipeline(cfg)
    pipeline.load_indexes()

    print(f"\n{'='*60}")
    print(f"QUESTION: {args.question}")
    print(f"{'='*60}\n")

    result = pipeline.run(
        question=args.question,
        image_path=args.image,
        log_snippet=args.log,
    )

    print("ANSWER:")
    print("-" * 40)
    print(result.answer)
    print()

    print("SOURCES:")
    print("-" * 40)
    for i, chunk in enumerate(result.selected_chunks, 1):
        print(f"[{i}] {chunk.get('title', 'Unknown')} — {chunk.get('section', '')}")
        print(f"    URL: {chunk.get('source_url', '')}")
        score = chunk.get('reranker_score', chunk.get('rrf_score', 0))
        print(f"    Score: {score:.4f}")
    print()

    if result.latency:
        print("TIMING:")
        for k, v in result.latency.to_dict().items():
            print(f"  {k}: {v}s")
    print()

    if result.reformulation and result.reformulation.was_reformulated:
        print("REFORMULATED QUERY:")
        print(f"  {result.reformulation.reformulated_query[:200]}")
        print()


if __name__ == "__main__":
    main()
