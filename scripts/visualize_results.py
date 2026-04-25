"""
Visualize evaluation results — generates the plots for the project report.

Reads JSON result files from the results/ directory and produces:
  - Retrieval comparison bar chart (BM25 vs dense vs hybrid)
  - Ablation table heatmap (system components)
  - Efficiency tradeoff scatter plot (tokens vs faithfulness)
  - Latency breakdown pie chart

Usage:
  python scripts/visualize_results.py --results results/evaluation.json
  python scripts/visualize_results.py --results-dir results/
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_retrieval_comparison(results_by_system: dict, output_path: str = "results/retrieval_comparison.png"):
    """Bar chart comparing Recall@k, MRR, nDCG@k across retrievers."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. pip install matplotlib")
        return

    metrics_of_interest = ["recall@5", "recall@10", "mrr", "ndcg@10"]
    systems = list(results_by_system.keys())
    x = np.arange(len(metrics_of_interest))
    width = 0.8 / len(systems)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, system in enumerate(systems):
        vals = [results_by_system[system].get(m, 0) for m in metrics_of_interest]
        bars = ax.bar(x + i * width, vals, width, label=system.replace("_", " ").title())

    ax.set_xticks(x + width * (len(systems) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics_of_interest])
    ax.set_ylabel("Score")
    ax.set_title("Retrieval System Comparison")
    ax.legend()
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_efficiency_tradeoff(ablation_rows: list, output_path: str = "results/efficiency_tradeoff.png"):
    """Scatter plot: prompt tokens (x) vs faithfulness (y), sized by latency."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for row in ablation_rows:
        if row.get("prompt_tokens", "—") == "—":
            continue
        x = float(row["prompt_tokens"])
        y = float(row.get("faithfulness", 0) or 0)
        size = float(row.get("latency_s", 1) or 1) * 200
        ax.scatter(x, y, s=size, alpha=0.7, label=row["setting"])
        ax.annotate(row["setting"], (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Average Prompt Tokens")
    ax.set_ylabel("Faithfulness Score")
    ax.set_title("Efficiency vs. Faithfulness Tradeoff\n(bubble size = latency)")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_latency_breakdown(latency_dict: dict, output_path: str = "results/latency_breakdown.png"):
    """Pie chart showing where time is spent in the pipeline."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    stage_labels = {
        "vlp_s": "VLP (screenshot parsing)",
        "retrieval_s": "Retrieval",
        "reranking_s": "Reranking",
        "pruning_s": "Context pruning",
        "generation_s": "Answer generation",
    }

    sizes = []
    labels = []
    for key, label in stage_labels.items():
        val = latency_dict.get(key, 0)
        if val and val > 0:
            sizes.append(val)
            labels.append(f"{label}\n({val:.2f}s)")

    if not sizes:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title("Pipeline Latency Breakdown")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--results", help="Path to a single evaluation JSON file")
    parser.add_argument("--retrieval-results", help="Path to retrieval comparison JSON")
    parser.add_argument("--ablation-results", help="Path to efficiency ablation JSON")
    parser.add_argument("--output-dir", default="results", help="Where to save plots")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.retrieval_results:
        data = load_results(args.retrieval_results)
        plot_retrieval_comparison(data, f"{args.output_dir}/retrieval_comparison.png")

    if args.ablation_results:
        data = load_results(args.ablation_results)
        plot_efficiency_tradeoff(data, f"{args.output_dir}/efficiency_tradeoff.png")

    if args.results:
        data = load_results(args.results)
        if "retrieval_metrics" in data:
            plot_retrieval_comparison(
                {"system": data["retrieval_metrics"]},
                f"{args.output_dir}/retrieval_metrics.png",
            )
        if "efficiency_metrics" in data:
            plot_latency_breakdown(
                data["efficiency_metrics"],
                f"{args.output_dir}/latency_breakdown.png",
            )

    print("Done.")


if __name__ == "__main__":
    main()
