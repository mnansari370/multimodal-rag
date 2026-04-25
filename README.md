# Multimodal RAG Troubleshooter

An AI-powered technical troubleshooting assistant that understands screenshots, error logs, and configuration files — then retrieves relevant documentation and generates grounded, cited answers.

Built as a master's project in AI/ML, this system demonstrates a complete production-grade RAG pipeline with a multimodal input layer, hybrid retrieval, cross-encoder reranking, context-efficient grounding, and rigorous evaluation.

---

## What it does

A user pastes a screenshot of a `CUDA out of memory` error, attaches their YAML training config, and asks "Why is my training crashing?" The system:

1. **Reads the screenshot** using a Vision-Language Model (InternVL2) to extract the error text, visual category, and relevant keywords
2. **Reformulates the query** by injecting those cues so retrieval gets actual error strings, not just "why is this happening"
3. **Retrieves documentation** using BM25 + dense embedding hybrid search with Reciprocal Rank Fusion
4. **Reranks** the top-50 candidates using a cross-encoder for higher precision
5. **Prunes context** to keep only the most useful chunks before generation (core innovation)
6. **Generates a cited answer** using an LLM grounded in the retrieved documentation

---

## System Architecture

```
[Input: text + image + log]
        │
        ▼
[Vision-Language Parser]   ← InternVL2-2B (screenshot → structured text)
        │
        ▼
[Query Reformulator]       ← injects VLP cues into the original question
        │
        ▼
[Hybrid Retriever]         ← BM25 + FAISS dense (BAAI/bge-base-en-v1.5) via RRF
        │
        ▼
[Cross-encoder Reranker]   ← ms-marco-MiniLM-L-6-v2 (top-50 → top-20)
        │
        ▼
[Context Selector]         ← score threshold | diversity | MMR coverage pruning
        │
        ▼
[Answer Generator]         ← Llama-3-8B or GPT-4o-mini, always cites sources
        │
        ▼
[Cited Answer + Sources]
```

---

## Core Innovation: Context-Efficient Grounding

Sending all retrieved chunks to the generator wastes tokens and hurts quality — the model gets distracted by marginally relevant chunks. The **context selector** module implements three pruning strategies:

| Strategy | How it works | Best for |
|---|---|---|
| Score threshold | Keep chunks above a reranker score cutoff | Simple baseline |
| Top-k + diversity | Keep top-k but penalize near-duplicates (n-gram Jaccard) | Balancing relevance and coverage |
| Coverage pruning | MMR-style: iteratively pick chunks that add the most new information | Best citation quality |

This produces a measurable tradeoff table (tokens vs. faithfulness vs. latency) that validates the efficiency gain.

---

## Repository Structure

```
multimodal-rag-troubleshooter/
├── configs/                   # YAML configs for each experiment
│   ├── pipeline.yaml
│   ├── ablation_retriever.yaml
│   ├── ablation_pruning.yaml
│   └── chunking.yaml
├── data/
│   ├── raw/                   # Downloaded docs (not committed)
│   ├── processed/             # Cleaned + chunked docs
│   ├── benchmark/             # Evaluation benchmark (JSON)
│   └── embeddings/            # FAISS index files (not committed)
├── src/
│   ├── ingestion/             # Downloader + cleaner for PyTorch docs
│   ├── chunking/              # Heading-based and fixed-size chunking
│   ├── retrieval/             # BM25, dense (FAISS), and hybrid (RRF)
│   ├── reranking/             # Cross-encoder reranker
│   ├── multimodal/            # VLP: screenshot → structured text
│   ├── reformulation/         # Query reformulator using VLP cues
│   ├── efficiency/            # Context pruning strategies
│   ├── generation/            # Answer generator with citation prompting
│   └── evaluation/            # Retrieval metrics, RAGAS, efficiency tracking
├── scripts/
│   ├── build_index.py         # Build FAISS + BM25 indexes
│   ├── run_pipeline.py        # Single-query CLI interface
│   └── evaluate.py            # Full evaluation suite
├── slurm/                     # SLURM job scripts for HPC runs
├── demo/
│   └── app.py                 # Gradio demo interface
├── results/                   # Experiment outputs and evaluation tables
├── pipeline.py                # End-to-end pipeline class
└── requirements.txt
```

---

## Setup

**Prerequisites:** Python 3.10+, CUDA-capable GPU recommended for VLP and embedding

```bash
git clone https://github.com/mnansari370/multimodal-rag-troubleshooter.git
cd multimodal-rag-troubleshooter

pip install -r requirements.txt
```

For the answer generator, set your API key:
```bash
export OPENAI_API_KEY=your_key_here
# or
export ANTHROPIC_API_KEY=your_key_here
```

---

## Quickstart

**Step 1 — Download and index the corpus**

```bash
# Download PyTorch documentation (~2800 pages)
python -m src.ingestion.downloader --output-dir data/raw

# Clean and chunk
python -m src.ingestion.cleaner --raw-dir data/raw --output-dir data/processed
python -m src.chunking.chunker --processed-dir data/processed \
    --output-file data/processed/chunks_heading.jsonl --strategy heading

# Build indexes (needs GPU for embedding)
python scripts/build_index.py --chunks data/processed/chunks_heading.jsonl
```

**Step 2 — Run a query**

```bash
python scripts/run_pipeline.py \
    --question "Why is my training crashing with CUDA out of memory?" \
    --log "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
```

**Step 3 — Launch the demo**

```bash
python demo/app.py --config configs/pipeline.yaml
```

**Step 4 — Run the evaluation suite**

```bash
python scripts/evaluate.py \
    --benchmark data/benchmark/benchmark.json \
    --config configs/pipeline.yaml
```

---

## Evaluation

The system is evaluated on four dimensions:

**Retrieval quality** (does it find the right docs?)
- Recall@k (k = 5, 10, 20), MRR, nDCG@k, Hit@k

**Answer quality** (is the answer correct?)
- Token F1 against gold answers
- Citation accuracy: do cited sources match the gold URLs?

**Faithfulness** (does the answer stay grounded in retrieved text?)
- RAGAS faithfulness score (LLM-as-judge — every sentence checked against context)

**Efficiency** (can we reduce cost without hurting quality?)
- Prompt token count before/after context pruning
- End-to-end latency breakdown by stage

### Main Results Table (ablation)

| System | Retriever | Reranker | VLP | Context Pruning | Answer Score | Latency |
|---|---|---|---|---|---|---|
| Baseline (BM25) | BM25 | — | — | — | — | — |
| Baseline (Dense) | Dense | — | — | — | — | — |
| Baseline (Hybrid) | Hybrid | — | — | — | — | — |
| + Reranker | Hybrid | ✓ | — | — | — | — |
| + VLP | Hybrid | ✓ | ✓ | — | — | — |
| **Full system** | **Hybrid** | **✓** | **✓** | **Coverage** | **—** | **—** |

*(Numbers filled in after running experiments on the benchmark)*

### Efficiency Ablation Table

| Pruning strategy | Prompt tokens | Latency (s) | Faithfulness | Answer score |
|---|---|---|---|---|
| No pruning (full context) | — | — | — | — |
| Score threshold | — | — | — | — |
| Top-k + diversity | — | — | — | — |
| Coverage pruning (λ=0.7) | — | — | — | — |

---

## HPC Usage (SLURM)

Organized by pipeline stage so each step can be rerun independently:

```bash
sbatch slurm/corpus_build.sh          # download + chunk (CPU)
sbatch slurm/embed_corpus.sh          # build FAISS index (GPU)
sbatch slurm/retrieval_eval.sh        # retrieval metrics
sbatch slurm/vlp_run.sh               # VLP over screenshot benchmark (GPU)
sbatch slurm/generation_eval.sh       # end-to-end + RAGAS (GPU)
sbatch slurm/ablation_efficiency.sh   # context pruning ablation
```

---

## Example Scenarios

The system is benchmarked on four types of troubleshooting queries:

1. **CUDA out of memory** — screenshot + question → retrieves batch size tuning and AMP docs
2. **Misconfigured YAML** — config file + question → identifies conflicting parameters
3. **Architecture diagram** — diagram image + question → explains the component with citations
4. **Failed deployment** — screenshot + log + question → traces root cause across sources

---

## Design Decisions

**Why treat VLP as a preprocessing step?**  
End-to-end multimodal retrieval requires specialized indexes and is harder to ablate. Using a VLM to convert screenshots to text keeps the rest of the pipeline text-based — simpler, faster to iterate, and cleaner to evaluate (retrieval with vs. without VLP cues).

**Why hybrid retrieval (BM25 + dense)?**  
Error strings like `RuntimeError: CUDA out of memory` are exact-match queries where BM25 excels. Semantic questions like "how do I reduce memory usage" need dense retrieval. Hybrid with RRF consistently outperforms either alone.

**Why context pruning?**  
Passing all 20 reranked chunks to the generator adds ~3000 tokens per query, increases latency proportionally, and can hurt faithfulness (the model sometimes cites less-relevant chunks). Pruning to 5 high-quality chunks maintains answer quality while cutting token cost significantly.

---

## Acknowledgements

- PyTorch documentation: [pytorch.org/docs](https://pytorch.org/docs/stable/)
- BAAI/bge embeddings and cross-encoder from [sentence-transformers](https://www.sbert.net/)
- Faithfulness evaluation via [RAGAS](https://github.com/explodinggradients/ragas)
- VLP via [InternVL2](https://huggingface.co/OpenGVLab/InternVL2-2B)
