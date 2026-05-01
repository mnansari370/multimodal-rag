"""
Query-mode ablation for multimodal RAG retrieval.

This experiment measures whether logs and VLP screenshot-derived text
actually improve retrieval.

Modes:
  1. query_only
  2. query_plus_log
  3. query_plus_vlp
  4. query_plus_log_plus_vlp

Output:
  results/query_mode_ablation.json
  results/tables/query_mode_ablation.md
"""

import sys
import json
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import MultimodalRAGPipeline, PipelineConfig
from src.evaluation.retrieval_metrics import evaluate_retrieval, print_retrieval_results


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def vlp_to_text(vlp: dict | None) -> str:
    """Flatten cached VLP output into retrieval text."""
    if not vlp or not isinstance(vlp, dict):
        return ""

    if "error" in vlp:
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
    """Build retrieval query according to ablation mode."""
    query = example.get("query", "").strip()
    log = example.get("log_snippet", "").strip()
    vlp_text = vlp_to_text(vlp_outputs.get(example.get("id")))

    parts = []

    if mode in {"query_only", "query_plus_log", "query_plus_vlp", "query_plus_log_plus_vlp"}:
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "query_only":
        parts = [query]

    elif mode == "query_plus_log":
        parts = [log, query]

    elif mode == "query_plus_vlp":
        parts = [vlp_text, query]

    elif mode == "query_plus_log_plus_vlp":
        parts = [vlp_text, log, query]

    return " ".join(p for p in parts if p).strip()


def main():
    parser = argparse.ArgumentParser(description="Run query-mode retrieval ablation")
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.json")
    parser.add_argument("--vlp-outputs", default="results/vlp_outputs.json")
    parser.add_argument("--output", default="results/query_mode_ablation.json")
    parser.add_argument("--retriever", default="hybrid", choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    benchmark_path = ROOT / args.benchmark
    vlp_path = ROOT / args.vlp_outputs
    output_path = ROOT / args.output

    logger.info("Loading benchmark: %s", benchmark_path)
    benchmark = json.load(open(benchmark_path, encoding="utf-8"))

    if vlp_path.exists():
        logger.info("Loading cached VLP outputs: %s", vlp_path)
        vlp_outputs = json.load(open(vlp_path, encoding="utf-8"))
    else:
        logger.warning("No VLP output file found. VLP modes will behave like query_only/log modes.")
        vlp_outputs = {}

    cfg = PipelineConfig(
        retriever_type=args.retriever,
        use_reranker=args.use_reranker,
        use_vlp=False,
    )

    pipe = MultimodalRAGPipeline(cfg)
    pipe.load_indexes()

    modes = [
        "query_only",
        "query_plus_log",
        "query_plus_vlp",
        "query_plus_log_plus_vlp",
    ]

    all_results = {}

    for mode in modes:
        logger.info("")
        logger.info("=" * 80)
        logger.info("Mode: %s", mode)
        logger.info("=" * 80)

        def retrieve_fn(_query_from_eval, mode=mode):
            # evaluate_retrieval passes only the raw query string, so we ignore it
            # and use a closure over the current example via wrapped benchmark below.
            raise RuntimeError("This function should not be called directly.")

        # Create a temporary benchmark where the query field is replaced
        # by the mode-specific effective query.
        mode_benchmark = []
        for ex in benchmark:
            ex2 = dict(ex)
            ex2["query"] = build_query(ex, mode, vlp_outputs)
            mode_benchmark.append(ex2)

        def mode_retrieve_fn(q):
            candidates = pipe.retriever.search(q, top_k=50)
            if args.use_reranker:
                return pipe.reranker.rerank(q, candidates, top_k=args.top_k)
            return candidates[:args.top_k]

        metrics = evaluate_retrieval(mode_benchmark, mode_retrieve_fn)
        print_retrieval_results(metrics, label=mode)
        all_results[mode] = metrics

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, open(output_path, "w", encoding="utf-8"), indent=2)
    logger.info("Saved results to %s", output_path)

    # Write markdown table
    table_path = ROOT / "results/tables/query_mode_ablation.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["Query mode", "MRR", "Recall@5", "Recall@10", "Recall@20", "Hit@10", "nDCG@10"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for mode, m in all_results.items():
        lines.append(
            "| "
            + " | ".join([
                mode,
                str(m.get("mrr")),
                str(m.get("recall@5")),
                str(m.get("recall@10")),
                str(m.get("recall@20")),
                str(m.get("hit@10")),
                str(m.get("ndcg@10")),
            ])
            + " |"
        )

    table_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved table to %s", table_path)


if __name__ == "__main__":
    main()
