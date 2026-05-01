"""
Vague-query VLP stress test.

This simulates realistic multimodal troubleshooting:
the user uploads a screenshot and asks a vague question like
"Why is this happening?"

We compare:
  1. original_explicit_query
  2. vague_query_only
  3. vague_query_plus_log
  4. vague_query_plus_vlp
  5. vague_query_plus_log_plus_vlp

This shows whether the VLP is useful when the typed question alone
does not contain the error terms.
"""

import sys
import json
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import MultimodalRAGPipeline, PipelineConfig
from src.evaluation.retrieval_metrics import evaluate_retrieval, print_retrieval_results


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


VAGUE_QUERY = "Why is this happening and how do I fix it?"


def vlp_to_text(vlp: dict | None) -> str:
    if not vlp or not isinstance(vlp, dict) or "error" in vlp:
        return ""

    parts = []

    if vlp.get("error_message"):
        parts.append(vlp["error_message"])

    if vlp.get("visual_category"):
        parts.append(vlp["visual_category"])

    comps = vlp.get("software_components") or []
    if comps:
        parts.append(" ".join(str(x) for x in comps))

    keywords = vlp.get("keywords") or []
    if keywords:
        parts.append(" ".join(str(x) for x in keywords))

    return " ".join(parts).strip()


def build_query(example: dict, mode: str, vlp_outputs: dict) -> str:
    original = example.get("query", "").strip()
    log = example.get("log_snippet", "").strip()
    vlp_text = vlp_to_text(vlp_outputs.get(example.get("id")))

    if mode == "original_explicit_query":
        return original

    if mode == "vague_query_only":
        return VAGUE_QUERY

    if mode == "vague_query_plus_log":
        return " ".join(x for x in [log, VAGUE_QUERY] if x).strip()

    if mode == "vague_query_plus_vlp":
        return " ".join(x for x in [vlp_text, VAGUE_QUERY] if x).strip()

    if mode == "vague_query_plus_log_plus_vlp":
        return " ".join(x for x in [vlp_text, log, VAGUE_QUERY] if x).strip()

    raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.json")
    parser.add_argument("--vlp-outputs", default="results/vlp_outputs.json")
    parser.add_argument("--output", default="results/vague_vlp_stress_test.json")
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    benchmark = json.load(open(ROOT / args.benchmark, encoding="utf-8"))
    vlp_outputs = json.load(open(ROOT / args.vlp_outputs, encoding="utf-8"))

    cfg = PipelineConfig(
        retriever_type="hybrid",
        use_reranker=args.use_reranker,
        use_vlp=False,
    )

    pipe = MultimodalRAGPipeline(cfg)
    pipe.load_indexes()

    modes = [
        "original_explicit_query",
        "vague_query_only",
        "vague_query_plus_log",
        "vague_query_plus_vlp",
        "vague_query_plus_log_plus_vlp",
    ]

    all_results = {}

    for mode in modes:
        logger.info("=" * 80)
        logger.info("Mode: %s", mode)
        logger.info("=" * 80)

        mode_benchmark = []
        for ex in benchmark:
            ex2 = dict(ex)
            ex2["query"] = build_query(ex, mode, vlp_outputs)
            mode_benchmark.append(ex2)

        def retrieve_fn(q):
            candidates = pipe.retriever.search(q, top_k=50)
            if args.use_reranker:
                return pipe.reranker.rerank(q, candidates, top_k=args.top_k)
            return candidates[:args.top_k]

        metrics = evaluate_retrieval(mode_benchmark, retrieve_fn)
        print_retrieval_results(metrics, label=mode)
        all_results[mode] = metrics

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, open(out, "w", encoding="utf-8"), indent=2)
    logger.info("Saved %s", out)

    table_path = ROOT / "results/tables/vague_vlp_stress_test.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "| Mode | MRR | Recall@5 | Recall@10 | Recall@20 | Hit@10 | nDCG@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, m in all_results.items():
        lines.append(
            f"| {mode} | {m.get('mrr')} | {m.get('recall@5')} | {m.get('recall@10')} | "
            f"{m.get('recall@20')} | {m.get('hit@10')} | {m.get('ndcg@10')} |"
        )

    table_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %s", table_path)


if __name__ == "__main__":
    main()
