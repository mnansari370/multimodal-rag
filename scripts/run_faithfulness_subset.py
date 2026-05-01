"""
Small LLM-as-judge faithfulness evaluation for the Multimodal RAG project.

This is intentionally lightweight:
  - runs the final RAG pipeline on a small subset of benchmark examples
  - asks Claude to judge whether the generated answer is supported by the retrieved context
  - saves per-example judgments + aggregate faithfulness/relevance/context scores

This is not official RAGAS. It is a controlled LLM-as-judge faithfulness evaluation.
"""

import os
import re
import json
import yaml
import time
import random
import argparse
import logging
from pathlib import Path

import anthropic

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import MultimodalRAGPipeline, PipelineConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


JUDGE_PROMPT = """You are evaluating a Retrieval-Augmented Generation answer.

Your task:
Decide whether the answer is faithful to the provided documentation context.

Definitions:
- Faithful means the answer's factual claims are supported by the context.
- Unsupported claims are claims that are not clearly present in the context.
- It is okay if the answer is incomplete, but it should not invent facts.
- Be strict about unsupported technical claims.

Return ONLY valid JSON with this exact schema:
{
  "faithfulness_score": 0.0,
  "answer_relevance_score": 0.0,
  "context_sufficiency_score": 0.0,
  "unsupported_claim_count": 0,
  "verdict": "pass | partial | fail",
  "short_reason": "one short sentence"
}

Scoring:
- faithfulness_score: 1.0 = all claims supported, 0.5 = partly supported, 0.0 = mostly unsupported
- answer_relevance_score: 1.0 = directly answers question, 0.0 = irrelevant
- context_sufficiency_score: 1.0 = context contains enough evidence, 0.0 = context is insufficient
"""


def load_config(path: str) -> PipelineConfig:
    cfg = PipelineConfig()
    p = Path(path)
    if p.exists():
        data = yaml.safe_load(open(p, encoding="utf-8")) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {
            "faithfulness_score": 0.0,
            "answer_relevance_score": 0.0,
            "context_sufficiency_score": 0.0,
            "unsupported_claim_count": -1,
            "verdict": "fail",
            "short_reason": "Judge did not return valid JSON.",
            "raw_judge_output": text,
        }

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {
            "faithfulness_score": 0.0,
            "answer_relevance_score": 0.0,
            "context_sufficiency_score": 0.0,
            "unsupported_claim_count": -1,
            "verdict": "fail",
            "short_reason": "Judge JSON could not be parsed.",
            "raw_judge_output": text,
        }


def build_context(chunks: list[dict], max_chars_per_chunk: int = 1200) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        title = c.get("title", "Unknown")
        section = c.get("section", "")
        url = c.get("source_url", "")
        text = c.get("text", "")
        text = text[:max_chars_per_chunk]
        parts.append(
            f"[Source {i}] {title} — {section}\n"
            f"URL: {url}\n"
            f"{text}"
        )
    return "\n\n---\n\n".join(parts)


def judge_answer(
    client,
    model: str,
    question: str,
    answer: str,
    context: str,
    max_retries: int = 3,
) -> tuple[dict, dict]:
    user_prompt = f"""Question:
{question}

Generated answer:
{answer}

Retrieved documentation context:
{context}

Evaluate faithfulness now.
"""

    for attempt in range(1, max_retries + 1):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=300,
                temperature=0,
                system=JUDGE_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            parsed = extract_json(msg.content[0].text)
            usage = {
                "judge_input_tokens": getattr(msg.usage, "input_tokens", 0),
                "judge_output_tokens": getattr(msg.usage, "output_tokens", 0),
            }
            return parsed, usage
        except Exception as e:
            if attempt == max_retries:
                return {
                    "faithfulness_score": 0.0,
                    "answer_relevance_score": 0.0,
                    "context_sufficiency_score": 0.0,
                    "unsupported_claim_count": -1,
                    "verdict": "fail",
                    "short_reason": f"Judge API failed: {e}",
                }, {"judge_input_tokens": 0, "judge_output_tokens": 0}
            time.sleep(2 * attempt)


def select_subset(benchmark: list[dict], max_examples: int, seed: int) -> list[dict]:
    """Deterministic balanced-ish sample across the whole benchmark."""
    rng = random.Random(seed)
    items = list(benchmark)
    rng.shuffle(items)
    return items[:max_examples]


def write_markdown_table(path: Path, aggregate: dict):
    text = f"""| Metric | Value |
| --- | ---: |
| Examples | {aggregate["num_examples"]} |
| Avg faithfulness | {aggregate["avg_faithfulness"]} |
| Avg answer relevance | {aggregate["avg_answer_relevance"]} |
| Avg context sufficiency | {aggregate["avg_context_sufficiency"]} |
| Pass rate | {aggregate["pass_rate"]} |
| Partial rate | {aggregate["partial_rate"]} |
| Fail rate | {aggregate["fail_rate"]} |
| Avg unsupported claims | {aggregate["avg_unsupported_claims"]} |
| Avg total latency seconds | {aggregate["avg_total_latency_s"]} |
| Approx judge input tokens | {aggregate["total_judge_input_tokens"]} |
| Approx judge output tokens | {aggregate["total_judge_output_tokens"]} |
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.json")
    parser.add_argument("--config", default="configs/pipeline_claude.yaml")
    parser.add_argument("--output", default="results/faithfulness_subset_20.json")
    parser.add_argument("--table", default="results/tables/faithfulness_subset_20.md")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--use-images", action="store_true",
                        help="If set, pass benchmark images to the pipeline. This adds VLP API calls.")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Run: set -a; source .env; set +a")

    benchmark = json.load(open(args.benchmark, encoding="utf-8"))
    subset = select_subset(benchmark, args.max_examples, args.seed)

    cfg = load_config(args.config)

    # Keep final system settings, but avoid image/VLP calls unless explicitly requested.
    # This evaluates faithfulness of generated answers grounded in retrieved context.
    if not args.use_images:
        cfg.use_vlp = False

    pipeline = MultimodalRAGPipeline(cfg)
    pipeline.load_indexes()

    client = anthropic.Anthropic(api_key=api_key)

    rows = []

    for i, ex in enumerate(subset, 1):
        ex_id = ex.get("id", f"ex_{i:03d}")
        question = ex["query"]
        log_snippet = ex.get("log_snippet", "")

        image_path = None
        if args.use_images:
            maybe_image = ex.get("image_path")
            if maybe_image and Path(maybe_image).exists():
                image_path = maybe_image

        logger.info("[%d/%d] %s | %s", i, len(subset), ex_id, question[:80])

        result = pipeline.run(
            question=question,
            image_path=image_path,
            log_snippet=log_snippet,
        )

        context = build_context(result.selected_chunks)

        judgment, usage = judge_answer(
            client=client,
            model=args.judge_model,
            question=question,
            answer=result.answer,
            context=context,
        )

        row = {
            "id": ex_id,
            "category": ex.get("category"),
            "difficulty": ex.get("difficulty"),
            "query": question,
            "used_image": bool(image_path),
            "answer": result.answer,
            "selected_source_urls": [c.get("source_url", "") for c in result.selected_chunks],
            "latency": result.latency.to_dict() if result.latency else {},
            "generator_prompt_tokens": result.generator_output.prompt_tokens if result.generator_output else 0,
            "generator_completion_tokens": result.generator_output.completion_tokens if result.generator_output else 0,
            "judgment": judgment,
            **usage,
        }
        rows.append(row)

        logger.info(
            "  faithfulness=%s relevance=%s verdict=%s reason=%s",
            judgment.get("faithfulness_score"),
            judgment.get("answer_relevance_score"),
            judgment.get("verdict"),
            judgment.get("short_reason"),
        )

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"examples": rows}, open(args.output, "w", encoding="utf-8"), indent=2)

    def avg(key):
        vals = [float(r["judgment"].get(key, 0.0)) for r in rows]
        return round(sum(vals) / max(len(vals), 1), 4)

    verdicts = [r["judgment"].get("verdict", "fail") for r in rows]
    unsupported = [
        max(0, int(r["judgment"].get("unsupported_claim_count", 0)))
        for r in rows
        if int(r["judgment"].get("unsupported_claim_count", 0)) >= 0
    ]

    total_latency = [
        float(r["latency"].get("total_s", 0.0))
        for r in rows
        if r.get("latency")
    ]

    aggregate = {
        "num_examples": len(rows),
        "seed": args.seed,
        "use_images": args.use_images,
        "judge_model": args.judge_model,
        "generator_model": cfg.generator_model,
        "avg_faithfulness": avg("faithfulness_score"),
        "avg_answer_relevance": avg("answer_relevance_score"),
        "avg_context_sufficiency": avg("context_sufficiency_score"),
        "pass_rate": round(verdicts.count("pass") / max(len(verdicts), 1), 4),
        "partial_rate": round(verdicts.count("partial") / max(len(verdicts), 1), 4),
        "fail_rate": round(verdicts.count("fail") / max(len(verdicts), 1), 4),
        "avg_unsupported_claims": round(sum(unsupported) / max(len(unsupported), 1), 3),
        "avg_total_latency_s": round(sum(total_latency) / max(len(total_latency), 1), 3),
        "total_judge_input_tokens": sum(int(r.get("judge_input_tokens", 0)) for r in rows),
        "total_judge_output_tokens": sum(int(r.get("judge_output_tokens", 0)) for r in rows),
    }

    final = {
        "aggregate": aggregate,
        "examples": rows,
    }

    json.dump(final, open(args.output, "w", encoding="utf-8"), indent=2)
    write_markdown_table(Path(args.table), aggregate)

    print("\n=== Faithfulness subset summary ===")
    print(json.dumps(aggregate, indent=2))
    print(f"\nSaved JSON:  {args.output}")
    print(f"Saved table: {args.table}")


if __name__ == "__main__":
    main()
