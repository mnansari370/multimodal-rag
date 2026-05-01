import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tables = ROOT / "results/tables"
tables.mkdir(parents=True, exist_ok=True)

def load(name):
    return json.load(open(ROOT / name, encoding="utf-8"))

def md_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)

# Retriever table
retr = load("results/retriever_ablation.json")
rows = []
for name, m in retr.items():
    rows.append([
        name,
        m.get("mrr"),
        m.get("recall@5"),
        m.get("recall@10"),
        m.get("recall@20"),
        m.get("hit@10"),
        m.get("ndcg@10"),
    ])
txt = md_table(["System", "MRR", "Recall@5", "Recall@10", "Recall@20", "Hit@10", "nDCG@10"], rows)
(tables / "retriever_ablation.md").write_text(txt)

# Chunking table
chunk = load("results/chunking_comparison.json")
rows = []
for name, m in chunk.items():
    rows.append([name, m.get("mrr"), m.get("recall@5"), m.get("recall@10"), m.get("recall@20"), m.get("hit@10")])
txt = md_table(["Chunking", "MRR", "Recall@5", "Recall@10", "Recall@20", "Hit@10"], rows)
(tables / "chunking_comparison.md").write_text(txt)

# Pruning table
prune = load("results/pruning_ablation.json")
rows = []
for r in prune:
    rows.append([
        r["setting"],
        r["avg_original_tokens"],
        r["avg_pruned_tokens"],
        r["avg_token_reduction_pct"],
        r["avg_selected_chunks"],
        r["avg_retrieve_rerank_prune_latency_s"],
    ])
txt = md_table(["Setting", "Original tokens", "Pruned tokens", "Reduction %", "Selected chunks", "Latency s"], rows)
(tables / "pruning_ablation.md").write_text(txt)

print("Wrote tables to results/tables/")
