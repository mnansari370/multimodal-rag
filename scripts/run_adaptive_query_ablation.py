"""
Adaptive query reformulation ablation.

This tests a practical strategy:
- If the user query is already specific, keep it mostly unchanged.
- If the query is vague, inject VLP/log cues.
- If the query is moderately specific, add only the first strong error line.

This is designed to avoid retrieval noise from blindly appending long logs.
"""

import sys
import json
import re
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import MultimodalRAGPipeline, PipelineConfig
from src.evaluation.retrieval_metrics import evaluate_retrieval, print_retrieval_results


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


ERROR_KEYWORDS = [
    "runtimeerror", "valueerror", "typeerror", "keyerror", "cuda", "oom",
    "nan", "error", "exception", "traceback", "failed", "nccl",
    "batchnorm", "gradscaler", "autocast", "dataloader", "torch.compile",
    "checkpoint", "optimizer", "scheduler", "dtype", "float16", "bfloat16",
]


VAGUE_PATTERNS = [
    r"\bwhy is this happening\b",
    r"\bwhat is wrong\b",
    r"\bhow do i fix this\b",
    r"\bwhy does this fail\b",
    r"\bwhat does this mean\b",
    r"\bhelp me debug\b",
]


def is_vague(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in VAGUE_PATTERNS) or len(q.split()) <= 5


def is_specific(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in ERROR_KEYWORDS)


def extract_error_lines(log_text: str, max_lines: int = 1) -> list[str]:
    lines = []
    for line in log_text.splitlines():
        s = line.strip()
        if not s:
            continue
        if any(k in s.lower() for k in ERROR_KEYWORDS):
            lines.append(s)
    return lines[:max_lines]


def vlp_to_text(vlp: dict | None, max_keywords: int = 6) -> str:
    if not vlp or not isinstance(vlp, dict) or "error" in vlp:
        return ""

    parts = []

    if vlp.get("error_message"):
        parts.append(vlp["error_message"])

    keywords = vlp.get("keywords") or []
    if keywords:
        parts.append(" ".join(str(x) for x in keywords[:max_keywords]))

    return " ".join(parts).strip()


def build_adaptive_query(example: dict, vlp_outputs: dict) -> str:
    query = example.get("query", "").strip()
    log = example.get("log_snippet", "").strip()
    vlp = vlp_to_text(vlp_outputs.get(example.get("id")))

    # Case 1: explicit query already contains useful terms.
    # Keep it clean to avoid noise.
    if is_specific(query) and not is_vague(query):
        return query

    # Case 2: vague query. Use all available context.
    if is_vague(query):
        return " ".join(x for x in [vlp, " ".join(extract_error_lines(log, 2)), query] if x).strip()

    # Case 3: moderate query. Add only first error line or concise VLP.
    error_lines = extract_error_lines(log, 1)
    return " ".join(x for x in [" ".join(error_lines), vlp, query] if x).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.json")
    parser.add_argument("--vlp-outputs", default="results/vlp_outputs.json")
    parser.add_argument("--output", default="results/adaptive_query_ablation.json")
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

    experiments = {}

    # Baseline: original query only
    query_only_benchmark = [dict(ex, query=ex["query"]) for ex in benchmark]
    experiments["query_only"] = query_only_benchmark

    # Adaptive reformulation
    adaptive_benchmark = []
    for ex in benchmark:
        ex2 = dict(ex)
        ex2["query"] = build_adaptive_query(ex, vlp_outputs)
        adaptive_benchmark.append(ex2)
    experiments["adaptive_query"] = adaptive_benchmark

    all_results = {}

    for name, bench in experiments.items():
        logger.info("=" * 80)
        logger.info("Experiment: %s", name)
        logger.info("=" * 80)

        def retrieve_fn(q):
            candidates = pipe.retriever.search(q, top_k=50)
            if args.use_reranker:
                return pipe.reranker.rerank(q, candidates, top_k=args.top_k)
            return candidates[:args.top_k]

        metrics = evaluate_retrieval(bench, retrieve_fn)
        print_retrieval_results(metrics, label=name)
        all_results[name] = metrics

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, open(output_path, "w", encoding="utf-8"), indent=2)
    logger.info("Saved %s", output_path)


if __name__ == "__main__":
    main()
