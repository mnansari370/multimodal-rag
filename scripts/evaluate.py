import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import MultimodalRAGPipeline, PipelineConfig
from src.evaluation import (
    evaluate_retrieval,
    evaluate_answers,
    compute_efficiency_stats,
    print_retrieval_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> PipelineConfig:
    cfg = PipelineConfig()
    p = Path(path)
    if p.exists():
        import yaml
        data = yaml.safe_load(open(p)) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multimodal RAG system")
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.json")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-ragas", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    benchmark = json.load(open(args.benchmark, encoding="utf-8"))
    if args.max_examples:
        benchmark = benchmark[: args.max_examples]

    logger.info(f"Running evaluation on {len(benchmark)} examples")

    cfg = load_config(args.config)
    pipeline = MultimodalRAGPipeline(cfg)
    pipeline.load_indexes()

    logger.info("Running retrieval evaluation...")
    retrieval_metrics = evaluate_retrieval(
        benchmark,
        lambda q: pipeline.retrieve_only(q, top_k=20),
    )
    print_retrieval_results(retrieval_metrics)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"results/evaluation_{timestamp}.json"

    if args.retrieval_only:
        result = {
            "timestamp": timestamp,
            "benchmark_size": len(benchmark),
            "mode": "retrieval_only",
            "config": cfg.__dict__,
            "retrieval_metrics": retrieval_metrics,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(output_path, "w", encoding="utf-8"), indent=2)
        logger.info(f"Saved retrieval-only results to {output_path}")
        return

    logger.info("Running end-to-end generation evaluation...")
    predictions = []
    all_chunks = []
    pruning_stats = []
    latencies = []

    for i, example in enumerate(benchmark, 1):
        logger.info(f"Example {i}/{len(benchmark)}: {example['query'][:80]}")
        result = pipeline.run(
            question=example["query"],
            image_path=example.get("image_path") if example.get("image_path") and Path(example["image_path"]).exists() else None,
            log_snippet=example.get("log_snippet", ""),
        )
        predictions.append(result.answer)
        all_chunks.append(result.selected_chunks)
        pruning_stats.append(result.pruning_stats)
        if result.latency:
            latencies.append(result.latency)

    answer_metrics = evaluate_answers(
        benchmark,
        predictions,
        all_chunks,
        run_ragas=not args.no_ragas,
    )
    efficiency_metrics = compute_efficiency_stats(pruning_stats, latencies)

    result = {
        "timestamp": timestamp,
        "benchmark_size": len(benchmark),
        "mode": "end_to_end",
        "config": cfg.__dict__,
        "retrieval_metrics": retrieval_metrics,
        "answer_metrics": answer_metrics,
        "efficiency_metrics": efficiency_metrics,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(output_path, "w", encoding="utf-8"), indent=2)
    logger.info(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
