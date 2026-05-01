# Multimodal RAG for Technical Troubleshooting

A retrieval-augmented generation system that accepts screenshots, error logs, and natural language questions together, then retrieves relevant PyTorch documentation and produces grounded, cited answers.

Built as a master's project in AI/ML. The core research question: **does parsing a screenshot with a Vision-Language Model improve retrieval for vague technical queries?** The answer turned out to be yes, but only for vague queries — explicit queries are better left alone.

---

## What it does

A user uploads a screenshot of a `CUDA out of memory` crash, pastes their training config, and asks "Why is this happening?" The system:

1. **Vision-Language Parser (VLP)** — runs a VLM on the screenshot once and extracts the error string, visual category, software components, and retrieval keywords
2. **Adaptive query reformulation** — if the typed question is vague, injects the VLP cues into the retrieval query; if the question already contains error terms, leaves it clean (adding noise hurts explicit queries)
3. **Hybrid retrieval** — BM25 + dense FAISS (BAAI/bge-base-en-v1.5) fused with Reciprocal Rank Fusion
4. **Cross-encoder reranking** — ms-marco-MiniLM-L-6-v2 scores each (query, chunk) pair; top-50 candidates → top-20
5. **Context pruning** — MMR-style coverage selection narrows 20 chunks to 5, cutting ~75% of prompt tokens while preserving faithfulness
6. **Answer generation** — Claude Haiku produces a structured answer with `[Source N]` citations tied to documentation URLs

---

## System Architecture

```
User input: question + (optional) screenshot + (optional) log/config
                │
                ▼
    ┌─────────────────────┐
    │  Vision-Language    │  Claude Haiku / GPT-4o-mini / InternVL2
    │  Parser (VLP)       │  → error_message, keywords, components
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Adaptive Query     │  explicit query → keep clean
    │  Reformulator       │  vague query   → prepend VLP + log cues
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Hybrid Retriever   │  BM25 (rank_bm25) + dense FAISS
    │  (BM25 + Dense)     │  fused via RRF (k=60)
    └──────────┬──────────┘    top-50 candidates
               │
               ▼
    ┌─────────────────────┐
    │  Cross-encoder      │  ms-marco-MiniLM-L-6-v2
    │  Reranker           │  top-50 → top-20
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Context Selector   │  MMR coverage pruning (λ=0.7)
    │  (core innovation)  │  top-20 → top-5 (~75% token reduction)
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Answer Generator   │  Claude Haiku
    │                     │  structured answer with [Source N] citations
    └─────────────────────┘
```

---

## Example Output

**Query:** `Why is my training crashing with CUDA out of memory?`  
**Log:** `RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has 10.76 GiB total; 8.45 GiB already allocated; 1.23 GiB free.`

```
Root cause:
- The training job is allocating more memory than the GPU can provide [Source 1].
  PyTorch's CUDA caching allocator holds reserved blocks, so the actual usable
  memory may be lower than total - allocated [Source 2].

Fix:
- Reduce batch size to lower the peak activation memory per step [Source 1].
- Enable mixed precision with torch.autocast('cuda') and GradScaler — fp16/bf16
  roughly halves activation memory [Source 3].
- For very large models, use gradient checkpointing:
  torch.utils.checkpoint.checkpoint(fn, *inputs) recomputes activations during
  backward instead of storing them [Source 2].

Why this works:
- Mixed precision stores activations in 16-bit during forward pass [Source 3].
  Gradient checkpointing trades compute for memory by discarding intermediate
  tensors and recomputing them on demand during backward [Source 2].

Sources used:
- [Source 1]: https://pytorch.org/docs/stable/notes/cuda.html
- [Source 2]: https://pytorch.org/docs/stable/checkpoint.html
- [Source 3]: https://pytorch.org/docs/stable/amp.html
```

---

## Benchmark

The evaluation uses **150 hand-crafted examples** covering 12 PyTorch error categories at three difficulty levels. Each example has a natural language query, a rendered screenshot (Pillow), an optional log/config snippet, a gold answer, and gold documentation URLs.

**Categories:** `cuda_memory`, `amp`, `dataloader`, `batch_norm`, `optimizer`, `dtype`, `distributed`, `architecture`, `config`, `checkpointing`, `torch_compile`, `general`

**Difficulty split:** 50 easy / 55 medium / 45 hard

**Corpus:** ~1,200 PyTorch documentation pages, chunked by section headings into ~17,000 chunks

---

## Results

### Retrieval ablation (150 examples)

All settings evaluated on the full 150-example benchmark. URLs are deduplicated before metric computation — multiple chunks from the same page count as one hit.

| System | MRR | Recall@5 | Recall@10 | Recall@20 | Hit@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 only | 0.363 | 0.377 | 0.419 | 0.451 | 0.640 | 0.321 |
| Dense only (bge-base-en-v1.5) | 0.411 | 0.416 | 0.478 | 0.522 | 0.680 | 0.365 |
| Hybrid (BM25 + Dense, RRF) | 0.458 | 0.448 | 0.510 | 0.536 | **0.747** | **0.406** |
| Hybrid + reranker | **0.475** | 0.428 | 0.501 | 0.506 | 0.733 | 0.403 |

The hybrid retriever consistently beats either component alone. The reranker improves MRR (it elevates the single best result) but slightly hurts Recall@20 and Hit@10 — it trades coverage for precision.

### Chunking strategy

| Strategy | MRR | Recall@5 | Recall@10 | Recall@20 | Hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heading-based | 0.458 | 0.448 | **0.510** | **0.536** | **0.747** |
| Fixed-size (400 tok, 20% overlap) | 0.463 | 0.435 | 0.479 | 0.513 | 0.680 |

Heading-based chunking wins on recall and Hit@10. Fixed-size gives slightly higher MRR (slightly better single-hit precision) but is weaker overall. The heading strategy is used in the final system.

### VLP impact: vague query stress test

This experiment replaces every benchmark query with `"Why is this happening and how do I fix it?"` — the worst-case vague question — and measures how much each input modality recovers retrieval.

| Input mode | MRR | Recall@10 | Hit@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Original explicit query (upper bound) | 0.475 | 0.501 | 0.733 | 0.403 |
| Vague query only | 0.030 | 0.180 | 0.300 | 0.059 |
| Vague + log snippet | 0.350 | 0.408 | 0.620 | 0.312 |
| **Vague + VLP screenshot cues** | **0.452** | **0.501** | **0.720** | **0.383** |
| Vague + log + VLP | 0.411 | 0.479 | 0.687 | 0.357 |

The key finding: VLP nearly fully recovers retrieval quality for vague queries — going from MRR 0.03 to 0.45, which matches the explicit query baseline (0.475). The log snippet alone helps but not nearly as much. Concatenating log + VLP is slightly worse than VLP alone, likely because the log text introduces noise that partially drowns out the structured VLP keywords.

### Context pruning efficiency (150 examples)

All experiments use the same hybrid + reranker stack. Only the context-selection step changes.

| Setting | Input chunks | Selected chunks | Original tokens | Pruned tokens | Reduction | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No pruning (all 20) | 20 | 20 | 7,094 | 7,094 | 0% | 1.80s |
| Score threshold (top-10) | 20 | ~4 | 7,094 | 1,417 | **80%** | 1.76s |
| Top-5 diversity (Jaccard) | 20 | 5 | 7,094 | 1,773 | 75% | 1.75s |
| Coverage pruning (λ=0.7) | 20 | 5 | 7,094 | 1,778 | 75% | 1.84s |
| Coverage pruning (λ=0.5) | 20 | 5 | 7,094 | 1,770 | 75% | 1.85s |

The threshold strategy is the most aggressive (80% reduction) but aggressively drops low-scoring chunks that may still be useful. The coverage (MMR) strategy consistently selects exactly 5 diverse chunks and is used in the final system because it produces better citation diversity in generated answers.

### Faithfulness evaluation (LLM-as-judge, 20 examples)

Claude Haiku judges whether each generated answer's claims are supported by the retrieved context.

| Metric | Value |
| --- | ---: |
| Avg faithfulness score | **0.923** |
| Avg answer relevance | **0.903** |
| Avg context sufficiency | 0.725 |
| Pass rate (fully faithful) | 65% |
| Partial rate (mostly faithful) | 35% |
| Fail rate (hallucination) | **0%** |
| Avg unsupported claims per answer | 0.6 |
| Avg total latency | 4.3s |

No answer was judged as a complete failure. The 35% partial rate reflects cases where the answer was correct but included a claim slightly beyond what the retrieved context explicitly stated.

### Full system performance (150 examples, end-to-end)

Final pipeline: hybrid retrieval + reranker + coverage pruning (top-5) + Claude Haiku

| Metric | Value |
| --- | ---: |
| Retrieval MRR | 0.458 |
| Retrieval Hit@10 | 0.747 |
| Retrieval nDCG@10 | 0.406 |
| Citation accuracy | 0.357 |
| Token F1 vs gold answers | 0.12 |
| Prompt token reduction | **75.3%** (7,311 → 1,781 avg) |
| Avg retrieval latency | 45ms |
| Avg generation latency | 4.24s |
| Avg total latency | 10.1s |

Token F1 is low (0.12) because the gold answers are short factual sentences while the generated answers are longer structured explanations — they cover the same content but in different words. Citation accuracy (0.36) reflects that not all cited pages are the primary gold source, though they are often related.

---

## Core Design Decisions

**Adaptive reformulation, not always augmentation**

Early experiments showed that appending the full log snippet to every query hurt MRR on explicit queries (MRR dropped from 0.475 to 0.416 for `query_plus_log`). The adaptive reformulator classifies each query: if it already contains technical terms (`RuntimeError`, `CUDA`, etc.), it is left unchanged. If it is vague (≤5 words or matches a vague pattern), VLP cues and the first error line are prepended. This design — inspired by the ablation data — is the difference between a helpful feature and retrieval noise.

**Why hybrid retrieval?**

BM25 is unbeatable for exact error strings: `RuntimeError: CUDA out of memory` retrieves the right CUDA docs immediately. Dense retrieval handles semantic questions like "how do I reduce memory during training" that don't contain the exact page's wording. Hybrid with RRF gives the best of both without tuning any score weights — rank position alone drives fusion.

**Why context pruning matters**

Without pruning, 20 reranked chunks → ~7,000 prompt tokens per query. That is expensive and empirically hurts answer quality: Claude sometimes cites marginally relevant chunk 15 over chunk 3. Pruning to 5 high-quality chunks is not just about cost — it constrains the generator to the most relevant evidence, which the faithfulness scores confirm (0.923 average with pruning).

**VLP as a preprocessing step, not retrieval integration**

End-to-end multimodal retrieval (e.g., CLIP indexes) would require a separate embedding pipeline and is harder to ablate cleanly. Treating VLP as a structured text extraction step keeps the rest of the pipeline fully text-based, which means every existing retrieval and evaluation tool works without modification.

---

## Repository Structure

```
.
├── pipeline.py                   # End-to-end PipelineConfig + MultimodalRAGPipeline
├── configs/
│   ├── pipeline.yaml             # Default config: hybrid + coverage pruning + Claude Haiku
│   └── pipeline_claude.yaml
├── src/
│   ├── ingestion/
│   │   ├── downloader.py         # Crawls PyTorch docs sitemap, extracts structured content
│   │   └── cleaner.py            # Filters boilerplate, normalizes whitespace
│   ├── chunking/
│   │   └── chunker.py            # Heading-based and fixed-size chunking strategies
│   ├── retrieval/
│   │   ├── bm25_retriever.py     # BM25Okapi wrapper with JSONL persistence
│   │   ├── dense_retriever.py    # FAISS flat IP index + bge-base-en-v1.5
│   │   └── hybrid_retriever.py   # RRF fusion of BM25 + dense
│   ├── reranking/
│   │   └── reranker.py           # CrossEncoder (ms-marco-MiniLM-L-6-v2)
│   ├── multimodal/
│   │   └── vlp.py                # VisionLanguageParser: Claude / GPT-4o-mini / InternVL2
│   ├── reformulation/
│   │   └── reformulator.py       # Adaptive query reformulation
│   ├── efficiency/
│   │   └── context_selector.py   # Threshold / diversity / MMR coverage pruning
│   ├── generation/
│   │   └── generator.py          # Anthropic API wrapper with citation prompting
│   └── evaluation/
│       ├── retrieval_metrics.py  # Recall@k, MRR, nDCG@k, Hit@k
│       ├── answer_metrics.py     # Token F1, citation accuracy, RAGAS
│       └── efficiency_metrics.py # Latency profiling, token cost tracking
├── scripts/
│   ├── ingest.py                 # Download + clean + chunk in one command
│   ├── build_index.py            # Build FAISS + BM25 indexes
│   ├── run_pipeline.py           # Single-query CLI
│   ├── evaluate.py               # Full benchmark evaluation
│   ├── run_retriever_ablation.py # BM25 vs dense vs hybrid vs +reranker
│   ├── run_pruning_ablation.py   # Threshold vs diversity vs coverage pruning
│   ├── run_query_mode_ablation.py # Query-only vs +log vs +VLP vs all
│   ├── run_adaptive_query_ablation.py # Adaptive vs naive query augmentation
│   ├── run_vague_vlp_stress_test.py   # VLP rescue of vague queries
│   ├── run_faithfulness_subset.py     # LLM-as-judge faithfulness evaluation
│   ├── run_vlp_batch.py          # Batch VLP over benchmark images (cacheable)
│   ├── run_chunking_comparison.py # Heading vs fixed-size chunking
│   ├── analyze_failures.py       # Find and classify retrieval failures
│   ├── human_eval.py             # Interactive CLI for human ratings
│   ├── visualize_results.py      # Matplotlib plots from results JSONs
│   ├── validate_data.py          # Sanity-check corpus and benchmark
│   ├── make_results_tables.py    # Convert result JSONs to markdown tables
│   └── build_benchmark_v2.py     # Generate 150-example benchmark + screenshots
├── demo/
│   └── app.py                    # Gradio interface
├── data/
│   ├── raw/                      # Downloaded doc pages (not committed)
│   ├── processed/                # Cleaned pages + chunk JSONL files
│   ├── embeddings/               # FAISS index + BM25 pickle (not committed)
│   └── benchmark/                # benchmark.json + rendered screenshots
├── results/
│   ├── tables/                   # Markdown tables for README
│   └── *.json                    # Raw evaluation outputs
├── slurm/                        # SLURM job scripts for HPC
└── requirements.txt
```

---

## Setup

**Prerequisites:** Python 3.10+, CUDA GPU recommended for embedding (CPU works but is slow)

```bash
git clone https://github.com/mnansari370/multimodal-rag-troubleshooter.git
cd multimodal-rag-troubleshooter
pip install -r requirements.txt
```

Set your API key for answer generation and VLP:
```bash
export ANTHROPIC_API_KEY=your_key_here
# GPT-4o-mini VLP backend (optional):
export OPENAI_API_KEY=your_key_here
```

---

## Quickstart

**1. Build the corpus and indexes**

```bash
# Download and chunk PyTorch docs (~1,200 pages, 30-60 min)
python scripts/ingest.py --heading-only

# Build BM25 and FAISS indexes
python scripts/build_index.py --chunks data/processed/chunks_heading.jsonl
```

**2. Run a single query**

```bash
python scripts/run_pipeline.py \
    --question "Why is my training crashing with CUDA out of memory?" \
    --log "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
```

With a screenshot:
```bash
python scripts/run_pipeline.py \
    --question "What is wrong?" \
    --image path/to/screenshot.png \
    --log "batch_size: 512\nprecision: float16"
```

**3. Launch the Gradio demo**

```bash
python demo/app.py --config configs/pipeline.yaml
# opens at http://localhost:7860
```

**4. Run the benchmark evaluation**

```bash
# First build the benchmark (generates 150 examples + screenshots)
python scripts/build_benchmark_v2.py

# Sanity-check everything is in order
python scripts/validate_data.py

# Run VLP over all benchmark screenshots (caches results for later)
python scripts/run_vlp_batch.py --backend anthropic --output results/vlp_outputs.json

# Full end-to-end evaluation
python scripts/evaluate.py --benchmark data/benchmark/benchmark.json
```

---

## Running the Ablations

Each ablation script is self-contained and saves results to `results/`:

```bash
# Retriever comparison: BM25 vs dense vs hybrid vs hybrid+reranker
python scripts/run_retriever_ablation.py

# Context pruning: threshold vs diversity vs MMR coverage
python scripts/run_pruning_ablation.py

# VLP impact: query-only vs +log vs +VLP vs all combined
python scripts/run_query_mode_ablation.py --vlp-outputs results/vlp_outputs.json

# Vague-query stress test (key VLP finding)
python scripts/run_vague_vlp_stress_test.py --vlp-outputs results/vlp_outputs.json

# Adaptive vs naive query reformulation
python scripts/run_adaptive_query_ablation.py --vlp-outputs results/vlp_outputs.json

# Faithfulness evaluation (LLM-as-judge, uses API)
python scripts/run_faithfulness_subset.py --max-examples 20

# Regenerate markdown tables from result JSONs
python scripts/make_results_tables.py
```

---

## HPC Usage (SLURM)

Jobs are organized by stage so each can be rerun independently:

```bash
sbatch slurm/corpus_build.sh          # download, clean, chunk (CPU node)
sbatch slurm/embed_corpus.sh          # build FAISS index (GPU node)
sbatch slurm/vlp_run.sh               # VLP over benchmark screenshots (GPU)
sbatch slurm/retrieval_eval.sh        # retriever ablation
sbatch slurm/ablation_efficiency.sh   # context pruning ablation
sbatch slurm/generation_eval.sh       # end-to-end + faithfulness
```

---

## Dependencies

Core stack:
- `sentence-transformers` — bge-base-en-v1.5 (dense retrieval) + ms-marco-MiniLM (reranking)
- `faiss-cpu` — approximate nearest-neighbor search
- `rank-bm25` — BM25Okapi sparse retrieval
- `anthropic` — Claude Haiku for VLP, generation, and LLM-as-judge
- `Pillow` — screenshot rendering for the benchmark
- `gradio` — demo interface

Optional:
- `ragas` — automated faithfulness / relevancy evaluation
- `transformers` + `torch` — InternVL2 local VLP backend (no API cost)

---

## Acknowledgements

- PyTorch documentation corpus: [pytorch.org/docs/stable](https://pytorch.org/docs/stable/)
- Embeddings: [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)
- Reranker: [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
- Faithfulness evaluation: [RAGAS](https://github.com/explodinggradients/ragas)
- Local VLP: [OpenGVLab/InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B)
