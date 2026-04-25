import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval import HybridRetriever
from src.evaluation.retrieval_metrics import evaluate_retrieval, print_retrieval_results

benchmark = json.load(open(ROOT / "data/benchmark/benchmark.json", encoding="utf-8"))

experiments = [
    ("heading", ROOT / "data/processed/chunks_heading.jsonl"),
    ("fixed", ROOT / "data/processed/chunks_fixed.jsonl"),
]

all_results = {}

for name, chunks_path in experiments:
    print("\n" + "=" * 80)
    print(f"Building and evaluating {name} chunks")
    print("=" * 80)

    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    retriever = HybridRetriever()
    retriever.build(chunks, show_progress=True)

    def retrieve_fn(query):
        return retriever.search(query, top_k=20)

    metrics = evaluate_retrieval(benchmark, retrieve_fn)
    print_retrieval_results(metrics, label=name)
    all_results[name] = metrics

with open(ROOT / "results/chunking_comparison_70.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print("\nSaved: results/chunking_comparison_70.json")
