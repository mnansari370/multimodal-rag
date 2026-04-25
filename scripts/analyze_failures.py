import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import MultimodalRAGPipeline, PipelineConfig
from src.evaluation.retrieval_metrics import _normalize_url

benchmark = json.load(open(ROOT / "data/benchmark/benchmark.json", encoding="utf-8"))

cfg = PipelineConfig(
    retriever_type="hybrid",
    use_reranker=True,
    use_vlp=False,
    generator_backend="local",
)
pipe = MultimodalRAGPipeline(cfg)
pipe.load_indexes()

failures = []

for ex in benchmark:
    query = (ex.get("log_snippet") or "") + " " + ex["query"]
    gold = {_normalize_url(u) for u in ex.get("gold_source_urls", [])}

    retrieved = pipe.retriever.search(query, top_k=50)
    reranked = pipe.reranker.rerank(query, retrieved, top_k=20)

    retrieved_urls = []
    for r in reranked[:10]:
        u = _normalize_url(r.get("source_url", ""))
        if u not in retrieved_urls:
            retrieved_urls.append(u)

    hit = bool(set(retrieved_urls) & gold)

    if not hit:
        failures.append({
            "id": ex["id"],
            "category": ex.get("category"),
            "difficulty": ex.get("difficulty"),
            "query": ex["query"],
            "gold_source_urls": ex.get("gold_source_urls", []),
            "top_retrieved_urls": retrieved_urls[:5],
            "likely_failure_type": "retriever found related but not exact source",
            "analysis_note": "Gold source was not in top-10 reranked URLs. Inspect whether query terms are too broad, chunking separated the relevant section, or reranker preferred a neighboring API page."
        })

out = ROOT / "results/failure_analysis_candidates.json"
json.dump(failures[:20], open(out, "w", encoding="utf-8"), indent=2)
print(f"Total failures found: {len(failures)}")
print(f"Saved top candidates: {out}")
