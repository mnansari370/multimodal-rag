#!/bin/bash
#SBATCH --job-name=ablation_efficiency
#SBATCH --output=results/logs/ablation_efficiency_%j.out
#SBATCH --error=results/logs/ablation_efficiency_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Context pruning ablation — produces the efficiency table
# Compares: no pruning | score threshold | diversity | coverage

cd $SLURM_SUBMIT_DIR

source activate DL_env

BENCHMARK=${1:-data/benchmark/benchmark.json}
OUTPUT=results/efficiency_ablation.json

python - <<EOF
import json
from pipeline import MultimodalRAGPipeline, PipelineConfig
from src.evaluation.efficiency_metrics import print_efficiency_table

with open("$BENCHMARK") as f:
    benchmark = json.load(f)

strategies = [
    {"name": "No pruning (all context)",  "pruning_strategy": "threshold", "pruning_threshold": -999, "context_top_k": 20},
    {"name": "Score threshold",            "pruning_strategy": "threshold", "pruning_threshold": 0.0,  "context_top_k": 10},
    {"name": "Top-k + diversity",          "pruning_strategy": "diversity", "context_top_k": 5},
    {"name": "Coverage pruning (λ=0.7)",  "pruning_strategy": "coverage",  "context_top_k": 5, "mmr_lambda": 0.7},
]

ablation_rows = []

for strategy_cfg in strategies:
    name = strategy_cfg.pop("name")
    cfg = PipelineConfig(**{k: v for k, v in strategy_cfg.items() if hasattr(PipelineConfig(), k)})
    pipeline = MultimodalRAGPipeline(cfg)
    pipeline.load_indexes()

    token_counts = []
    latencies = []

    for example in benchmark[:30]:  # run on 30 examples for speed
        result = pipeline.run(question=example["query"], log_snippet=example.get("log_snippet", ""))
        token_counts.append(result.pruning_stats.get("pruned_tokens", 0))
        latencies.append(result.latency.total_time_s if result.latency else 0.0)

    row = {
        "setting": name,
        "prompt_tokens": round(sum(token_counts) / max(len(token_counts), 1), 1),
        "latency_s": round(sum(latencies) / max(len(latencies), 1), 2),
        "faithfulness": "—",   # fill in after RAGAS run
        "answer_score": "—",   # fill in after answer evaluation
    }
    ablation_rows.append(row)
    print(f"Done: {name} | avg tokens: {row['prompt_tokens']} | avg latency: {row['latency_s']}s")

print_efficiency_table(ablation_rows)

with open("$OUTPUT", "w") as f:
    json.dump(ablation_rows, f, indent=2)

print(f"\nSaved to $OUTPUT")
EOF

echo "Efficiency ablation complete."
