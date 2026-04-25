import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import MultimodalRAGPipeline, PipelineConfig
from src.efficiency import select_context

benchmark = json.load(open(ROOT / "data/benchmark/benchmark.json", encoding="utf-8"))

experiments = [
    ("full_context_20", "threshold", -999, 20, 0.7),
    ("threshold_top10", "threshold", 0.0, 10, 0.7),
    ("top5_diversity", "diversity", 0.0, 5, 0.7),
    ("coverage_mmr_07", "coverage", 0.0, 5, 0.7),
    ("coverage_mmr_05", "coverage", 0.0, 5, 0.5),
]

cfg = PipelineConfig(
    retriever_type="hybrid",
    use_reranker=True,
    use_vlp=False,
    generator_backend="local",
)

pipeline = MultimodalRAGPipeline(cfg)
pipeline.load_indexes()

rows = []

for name, strategy, threshold, top_k, mmr_lambda in experiments:
    print("\n" + "=" * 80)
    print("Running:", name)
    print("=" * 80)

    total_input_chunks = 0
    total_selected_chunks = 0
    total_original_tokens = 0
    total_pruned_tokens = 0
    total_reduction = 0
    total_latency = 0

    for ex in benchmark:
        query = (ex.get("log_snippet") or "") + " " + ex["query"]

        t0 = time.time()
        retrieved = pipeline.retriever.search(query, top_k=50)
        reranked = pipeline.reranker.rerank(query, retrieved, top_k=20)

        selected, stats = select_context(
            reranked,
            strategy=strategy,
            query=query,
            threshold=threshold,
            top_k=top_k,
            lambda_param=mmr_lambda,
        )
        elapsed = time.time() - t0

        total_input_chunks += stats["input_chunks"]
        total_selected_chunks += stats["selected_chunks"]
        total_original_tokens += stats["original_tokens"]
        total_pruned_tokens += stats["pruned_tokens"]
        total_reduction += stats["token_reduction_pct"]
        total_latency += elapsed

    n = len(benchmark)
    row = {
        "setting": name,
        "strategy": strategy,
        "top_k": top_k,
        "threshold": threshold,
        "mmr_lambda": mmr_lambda,
        "avg_input_chunks": round(total_input_chunks / n, 2),
        "avg_selected_chunks": round(total_selected_chunks / n, 2),
        "avg_original_tokens": round(total_original_tokens / n, 2),
        "avg_pruned_tokens": round(total_pruned_tokens / n, 2),
        "avg_token_reduction_pct": round(total_reduction / n, 2),
        "avg_retrieval_rerank_prune_latency_s": round(total_latency / n, 3),
    }
    rows.append(row)

    print(json.dumps(row, indent=2))

out = ROOT / "results/pruning_ablation_70.json"
json.dump(rows, open(out, "w", encoding="utf-8"), indent=2)
print("\nSaved:", out)
