"""
Efficiency evaluation — token cost and latency tracking across pipeline runs.
"""

import time
import logging
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class LatencyProfile:
    """Tracks timing for each stage of the pipeline."""
    vlp_time_s: float = 0.0
    reformulation_time_s: float = 0.0
    retrieval_time_s: float = 0.0
    reranking_time_s: float = 0.0
    pruning_time_s: float = 0.0
    generation_time_s: float = 0.0

    @property
    def total_time_s(self) -> float:
        return (
            self.vlp_time_s
            + self.reformulation_time_s
            + self.retrieval_time_s
            + self.reranking_time_s
            + self.pruning_time_s
            + self.generation_time_s
        )

    def to_dict(self) -> dict:
        return {
            "vlp_s": round(self.vlp_time_s, 3),
            "reformulation_s": round(self.reformulation_time_s, 3),
            "retrieval_s": round(self.retrieval_time_s, 3),
            "reranking_s": round(self.reranking_time_s, 3),
            "pruning_s": round(self.pruning_time_s, 3),
            "generation_s": round(self.generation_time_s, 3),
            "total_s": round(self.total_time_s, 3),
        }


class Timer:
    """Simple context manager for timing a block of code."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def compute_efficiency_stats(
    pruning_stats_list: list[dict],
    latency_profiles: list[LatencyProfile],
) -> dict:
    """
    Aggregate efficiency metrics across a benchmark run.

    Args:
        pruning_stats_list: List of stats dicts from context_selector.select_context().
        latency_profiles: One LatencyProfile per benchmark example.

    Returns:
        Averaged efficiency stats dict.
    """
    n = max(len(pruning_stats_list), 1)

    avg_original_tokens = sum(s["original_tokens"] for s in pruning_stats_list) / n
    avg_pruned_tokens = sum(s["pruned_tokens"] for s in pruning_stats_list) / n
    avg_reduction = sum(s["token_reduction_pct"] for s in pruning_stats_list) / n

    avg_total_latency = sum(p.total_time_s for p in latency_profiles) / max(len(latency_profiles), 1)
    avg_generation_latency = sum(p.generation_time_s for p in latency_profiles) / max(len(latency_profiles), 1)
    avg_retrieval_latency = sum(p.retrieval_time_s for p in latency_profiles) / max(len(latency_profiles), 1)

    return {
        "avg_original_prompt_tokens": round(avg_original_tokens, 1),
        "avg_pruned_prompt_tokens": round(avg_pruned_tokens, 1),
        "avg_token_reduction_pct": round(avg_reduction, 1),
        "avg_total_latency_s": round(avg_total_latency, 3),
        "avg_generation_latency_s": round(avg_generation_latency, 3),
        "avg_retrieval_latency_s": round(avg_retrieval_latency, 3),
    }


def print_efficiency_table(rows: list[dict]):
    """Print the ablation table in a clean tabular format."""
    headers = ["Setting", "Prompt Tokens", "Latency (s)", "Faithfulness", "Ans. Score"]
    col_widths = [25, 15, 12, 14, 12]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print("\n" + "=" * len(header_line))
    print("EFFICIENCY ABLATION TABLE")
    print("=" * len(header_line))
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        values = [
            str(row.get("setting", "—")).ljust(col_widths[0]),
            str(row.get("prompt_tokens", "—")).ljust(col_widths[1]),
            str(row.get("latency_s", "—")).ljust(col_widths[2]),
            str(row.get("faithfulness", "—")).ljust(col_widths[3]),
            str(row.get("answer_score", "—")).ljust(col_widths[4]),
        ]
        print(" | ".join(values))

    print("=" * len(header_line) + "\n")
