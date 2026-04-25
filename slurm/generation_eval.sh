#!/bin/bash
#SBATCH --job-name=generation_eval
#SBATCH --output=results/logs/generation_eval_%j.out
#SBATCH --error=results/logs/generation_eval_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# End-to-end generation evaluation with RAGAS faithfulness scoring

cd $SLURM_SUBMIT_DIR

source activate DL_env

BENCHMARK=${1:-data/benchmark/benchmark.json}
CONFIG=${2:-configs/pipeline.yaml}
OUTPUT=results/generation_results.json

python - <<EOF
import json
from pipeline import MultimodalRAGPipeline, PipelineConfig

with open("$BENCHMARK") as f:
    benchmark = json.load(f)

cfg = PipelineConfig()
pipeline = MultimodalRAGPipeline(cfg)
pipeline.load_indexes()

predictions = []
all_chunks = []

for example in benchmark:
    result = pipeline.run(
        question=example["query"],
        log_snippet=example.get("log_snippet", ""),
    )
    predictions.append(result.answer)
    all_chunks.append(result.selected_chunks)
    print(f"Done: {example['query'][:50]}")

from src.evaluation import evaluate_answers
metrics = evaluate_answers(benchmark, predictions, all_chunks, run_ragas=True)

with open("$OUTPUT", "w") as f:
    json.dump({"metrics": metrics}, f, indent=2)

print("\nFinal metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")
EOF

echo "Generation evaluation complete."
