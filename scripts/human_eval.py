"""
Human evaluation tool — interactive CLI for rating generated answers.

Runs through benchmark examples, shows the system's answer, and prompts
you to rate it on four dimensions (1-5 scale). Saves ratings to a JSON file.

Run on 20-30 examples to produce credible human eval numbers for the report.

Usage:
  python scripts/human_eval.py \
    --benchmark data/benchmark/benchmark.json \
    --config configs/pipeline.yaml \
    --output results/human_eval.json \
    --num-examples 25
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import MultimodalRAGPipeline, PipelineConfig


DIMENSIONS = {
    "correctness": "Is the answer factually correct?",
    "usefulness":  "Does it actually help solve the problem?",
    "source_quality": "Are the cited sources relevant to the answer?",
    "clarity": "Is the answer clearly written and well-organized?",
}


def get_rating(prompt: str) -> int:
    while True:
        try:
            val = input(f"{prompt} [1-5]: ").strip()
            rating = int(val)
            if 1 <= rating <= 5:
                return rating
            print("  Please enter a number between 1 and 5.")
        except (ValueError, KeyboardInterrupt):
            print("  Please enter a number between 1 and 5.")


def main():
    parser = argparse.ArgumentParser(description="Human evaluation tool")
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.json")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-examples", type=int, default=25,
                        help="How many benchmark examples to rate")
    args = parser.parse_args()

    with open(args.benchmark) as f:
        benchmark = json.load(f)

    benchmark = benchmark[:args.num_examples]

    cfg = PipelineConfig()
    if Path(args.config).exists():
        import yaml
        with open(args.config) as f:
            data = yaml.safe_load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    pipeline = MultimodalRAGPipeline(cfg)
    pipeline.load_indexes()

    ratings = []
    print(f"\nHuman Evaluation — {len(benchmark)} examples")
    print("Rate each answer on a 1-5 scale (1=very poor, 5=excellent)\n")

    for i, example in enumerate(benchmark, 1):
        print("\n" + "="*70)
        print(f"Example {i}/{len(benchmark)}: [{example.get('category', '?')}] {example.get('difficulty', '?')}")
        print(f"QUERY: {example['query']}")
        if example.get("log_snippet"):
            print(f"LOG:   {example['log_snippet'][:150]}")
        print("-"*70)

        result = pipeline.run(
            question=example["query"],
            log_snippet=example.get("log_snippet", ""),
        )

        print(f"ANSWER:\n{result.answer}\n")
        print(f"SOURCES ({len(result.selected_chunks)} chunks):")
        for j, chunk in enumerate(result.selected_chunks[:3], 1):
            print(f"  [{j}] {chunk.get('title')} — {chunk.get('section')}")
            print(f"       {chunk.get('source_url', '')}")

        if example.get("gold_answer"):
            print(f"\nGOLD ANSWER (reference):\n{example['gold_answer'][:300]}")

        print("\nRatings:")
        example_ratings = {"id": example.get("id", f"ex_{i:03d}"), "query": example["query"]}
        for dim, question in DIMENSIONS.items():
            example_ratings[dim] = get_rating(f"  {question}")

        notes = input("  Notes (optional, press Enter to skip): ").strip()
        if notes:
            example_ratings["notes"] = notes

        ratings.append(example_ratings)
        print(f"  Saved. ({i}/{len(benchmark)} done)")

    # Compute averages
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for dim in DIMENSIONS:
        vals = [r[dim] for r in ratings if dim in r]
        avg = sum(vals) / max(len(vals), 1)
        print(f"  {dim:<20} avg={avg:.2f}/5.0  ({len(vals)} ratings)")

    overall = [sum(r[d] for d in DIMENSIONS if d in r) / len(DIMENSIONS) for r in ratings]
    print(f"\n  Overall avg:         {sum(overall)/max(len(overall),1):.2f}/5.0")

    output_path = args.output or f"results/human_eval_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_examples": len(ratings),
            "ratings": ratings,
            "averages": {
                dim: round(sum(r[dim] for r in ratings if dim in r) / max(len(ratings), 1), 3)
                for dim in DIMENSIONS
            },
        }, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
