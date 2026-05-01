"""
Convert evaluation JSON files into markdown tables for the README.

Reads results from results/ and writes formatted tables to results/tables/.

Usage:
    python scripts/make_results_tables.py
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "results/tables"


def load_json(name):
    return json.load(open(ROOT / name, encoding="utf-8"))


def md_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    retr = load_json("results/retriever_ablation.json")
    rows = [
        [
            name,
            m.get("mrr"),
            m.get("recall@5"),
            m.get("recall@10"),
            m.get("recall@20"),
            m.get("hit@10"),
            m.get("ndcg@10"),
        ]
        for name, m in retr.items()
    ]
    txt = md_table(["System", "MRR", "Recall@5", "Recall@10", "Recall@20", "Hit@10", "nDCG@10"], rows)
    (TABLES_DIR / "retriever_ablation.md").write_text(txt)

    chunk = load_json("results/chunking_comparison.json")
    rows = [
        [name, m.get("mrr"), m.get("recall@5"), m.get("recall@10"), m.get("recall@20"), m.get("hit@10")]
        for name, m in chunk.items()
    ]
    txt = md_table(["Chunking", "MRR", "Recall@5", "Recall@10", "Recall@20", "Hit@10"], rows)
    (TABLES_DIR / "chunking_comparison.md").write_text(txt)

    prune = load_json("results/pruning_ablation.json")
    rows = [
        [
            r["setting"],
            r["avg_original_tokens"],
            r["avg_pruned_tokens"],
            r["avg_token_reduction_pct"],
            r["avg_selected_chunks"],
            r["avg_retrieve_rerank_prune_latency_s"],
        ]
        for r in prune
    ]
    txt = md_table(
        ["Setting", "Original tokens", "Pruned tokens", "Reduction %", "Selected chunks", "Latency s"],
        rows,
    )
    (TABLES_DIR / "pruning_ablation.md").write_text(txt)

    print(f"Wrote tables to {TABLES_DIR}/")


if __name__ == "__main__":
    main()
