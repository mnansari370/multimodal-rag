import json
from pathlib import Path
from collections import Counter

def count_json(path):
    return len(list(Path(path).glob("*.json")))

def count_jsonl(path):
    p = Path(path)
    if not p.exists():
        return 0
    with open(p, encoding="utf-8") as f:
        return sum(1 for _ in f)

print("=== Corpus ===")
print("raw pages:", count_json("data/raw"))
print("processed pages:", len(list(Path("data/processed").glob("page_*.json"))))
print("heading chunks:", count_jsonl("data/processed/chunks_heading.jsonl"))
print("fixed chunks:", count_jsonl("data/processed/chunks_fixed.jsonl"))
print("embedding chunks:", count_jsonl("data/embeddings/chunks.jsonl"))

print("\n=== Index files ===")
for f in ["data/embeddings/bm25.pkl", "data/embeddings/dense.faiss", "data/embeddings/chunks.jsonl"]:
    p = Path(f)
    print(f, "OK" if p.exists() and p.stat().st_size > 0 else "MISSING/EMPTY")

print("\n=== Benchmark ===")
bench_path = Path("data/benchmark/benchmark.json")
if not bench_path.exists():
    print("benchmark.json missing")
    raise SystemExit(1)

data = json.load(open(bench_path, encoding="utf-8"))
print("examples:", len(data))
print("categories:", Counter(x.get("category") for x in data))
print("difficulties:", Counter(x.get("difficulty") for x in data))
print("with image_path:", sum(1 for x in data if x.get("image_path")))
print("existing images:", sum(1 for x in data if x.get("image_path") and Path(x["image_path"]).exists()))
print("with logs/configs:", sum(1 for x in data if x.get("log_snippet")))
print("with gold answers:", sum(1 for x in data if x.get("gold_answer")))
print("with gold URLs:", sum(1 for x in data if x.get("gold_source_urls")))

missing = []
for ex in data:
    img = ex.get("image_path")
    if img and not Path(img).exists():
        missing.append((ex.get("id"), img))

if missing:
    print("\nMissing images:")
    for ex_id, img in missing[:30]:
        print(f"{ex_id}: {img}")
else:
    print("\nNo missing image paths.")

if len(data) < 70:
    print("\nWARNING: benchmark has fewer than 70 examples. Report target is 70–100.")

if sum(1 for x in data if x.get("image_path") and Path(x["image_path"]).exists()) == 0:
    print("WARNING: no real benchmark images found. Multimodal evaluation is incomplete.")
