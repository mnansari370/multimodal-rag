import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import MultimodalRAGPipeline, PipelineConfig
from src.evaluation.retrieval_metrics import evaluate_retrieval, print_retrieval_results

benchmark = json.load(open(ROOT / "data/benchmark/benchmark.json", encoding="utf-8"))

experiments = [
    ("bm25", "bm25", False),
    ("dense", "dense", False),
    ("hybrid", "hybrid", False),
    ("hybrid_reranker", "hybrid", True),
]

all_results = {}

for name, retriever_type, use_reranker in experiments:
    print("\n" + "="*80)
    print("Running:", name)
    print("="*80)

    cfg = PipelineConfig(
        retriever_type=retriever_type,
        generator_backend="local",
        use_vlp=False,
    )

    pipe = MultimodalRAGPipeline(cfg)
    pipe.load_indexes()

    def retrieve_fn(query):
        res = pipe.retriever.search(query, top_k=50)
        if use_reranker:
            res = pipe.reranker.rerank(query, res, top_k=20)
        return res[:20]

    metrics = evaluate_retrieval(benchmark, retrieve_fn)
    print_retrieval_results(metrics, label=name)
    all_results[name] = metrics

with open(ROOT / "results/retriever_ablation_70.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nSaved results/retriever_ablation_70.json")
