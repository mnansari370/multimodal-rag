#!/bin/bash
#SBATCH --job-name=vlp_run
#SBATCH --output=results/logs/vlp_run_%j.out
#SBATCH --error=results/logs/vlp_run_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Run the VLP over all screenshot benchmark examples
# and save the structured descriptions to disk for reuse

cd $SLURM_SUBMIT_DIR

source activate DL_env

VLP_BACKEND=${1:-internvl2}
BENCHMARK=${2:-data/benchmark/benchmark.json}
OUTPUT=results/vlp_outputs.json

python - <<EOF
import json
from pathlib import Path
from PIL import Image
from src.multimodal import VisionLanguageParser

with open("$BENCHMARK") as f:
    benchmark = json.load(f)

vlp = VisionLanguageParser(backend="$VLP_BACKEND")
outputs = {}

for example in benchmark:
    qid = example.get("id", example["query"][:30])
    img_path = example.get("image_path", "")

    if img_path and Path(img_path).exists():
        result = vlp.parse(image_path=img_path)
        outputs[qid] = result.to_dict()
        print(f"Processed: {qid} → {result.visual_category}")
    else:
        outputs[qid] = None

with open("$OUTPUT", "w") as f:
    json.dump(outputs, f, indent=2)

print(f"Saved VLP outputs to $OUTPUT")
EOF

echo "VLP run complete."
