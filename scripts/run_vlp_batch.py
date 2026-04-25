import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.multimodal import VisionLanguageParser

bench = json.load(open(ROOT / "data/benchmark/benchmark.json", encoding="utf-8"))
out_path = ROOT / "results/vlp_outputs.json"

vlp = VisionLanguageParser(backend="internvl2")
outputs = {}

for ex in bench:
    ex_id = ex["id"]
    img = ex.get("image_path")
    if img and Path(img).exists():
        print(f"Processing {ex_id}: {img}")
        result = vlp.parse(image_path=img)
        outputs[ex_id] = result.to_dict()
    else:
        outputs[ex_id] = None

json.dump(outputs, open(out_path, "w", encoding="utf-8"), indent=2)
print(f"Saved: {out_path}")
