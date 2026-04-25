import json
from pathlib import Path

MAX_CHARS = 1600
OVERLAP = 320

processed = Path("data/processed")
out_file = processed / "chunks_fixed.jsonl"

def make_chunk(page, text, idx):
    return {
        "chunk_id": f"{Path(page['url']).name}_{idx}",
        "source_url": page["url"],
        "title": page.get("title", ""),
        "section": page.get("title", ""),
        "text": text.strip(),
        "approx_tokens": max(1, len(text) // 4),
        "strategy": "fixed",
        "chunk_idx": idx,
    }

count = 0
with open(out_file, "w", encoding="utf-8") as out:
    for fp in sorted(processed.glob("page_*.json")):
        page = json.load(open(fp, encoding="utf-8"))
        parts = [page.get("title", "")]
        parts += page.get("paragraphs", [])
        for cb in page.get("code_blocks", []):
            if cb.get("context"):
                parts.append(cb["context"])
            parts.append("```\n" + cb.get("code", "") + "\n```")
        for n in page.get("notes", []):
            parts.append(f"[{n.get('type','NOTE')}] {n.get('text','')}")
        text = "\n\n".join(p for p in parts if p).strip()
        if not text:
            continue

        start = 0
        idx = 0
        while start < len(text):
            end = min(start + MAX_CHARS, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                out.write(json.dumps(make_chunk(page, chunk_text, idx), ensure_ascii=False) + "\n")
                count += 1
            if end >= len(text):
                break
            start = max(end - OVERLAP, start + 1)
            idx += 1

print(f"Wrote {count} fixed chunks to {out_file}")
