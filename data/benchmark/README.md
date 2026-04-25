# Benchmark — How to Build It

The evaluation benchmark is manually curated. Each example has:
- `id`: unique identifier
- `query`: the user's natural-language question
- `image_path`: path to screenshot (optional, null if text-only)
- `log_snippet`: pasted log or config text (optional)
- `gold_answer`: a correct, concise reference answer
- `gold_source_urls`: list of PyTorch doc URLs that contain the answer
- `gold_chunk_ids`: filled in after indexing (can be left empty initially)
- `category`: error type — cuda_memory | batch_norm | dataloader | amp | optimizer | etc.
- `difficulty`: easy | medium | hard

## Sources for examples

**From real errors (best examples):**
- Run intentionally broken PyTorch code and screenshot the error
- Try invalid batch sizes, wrong dtypes, conflicting parameters
- Record both the error message and what fixed it

**From public resources:**
- PyTorch GitHub issues where the resolution linked to documentation
- Stack Overflow questions tagged [pytorch] where accepted answers cite docs
- HuggingFace forums for Transformers-related issues

**Minimum benchmark size:** 70 examples  
**Target for credible evaluation:** 100 examples

## Category distribution to aim for

| Category | Description | Target count |
|---|---|---|
| cuda_memory | CUDA OOM errors, memory management | 15 |
| amp | Mixed precision, autocast, GradScaler | 10 |
| dataloader | DataLoader workers, shared memory | 10 |
| batch_norm | BatchNorm with small batches | 8 |
| optimizer | Gradient clipping, learning rate schedulers | 10 |
| distributed | DDP, multiprocessing errors | 8 |
| dtype | Float/Half/BFloat16 mismatches | 8 |
| architecture | Model component questions (diagram-based) | 8 |
| config | YAML/config file misconfigurations | 8 |
| general | Everything else | 15 |

## Benchmark format

```json
[
  {
    "id": "ex_001",
    "query": "Why is my training crashing with CUDA out of memory?",
    "image_path": "data/benchmark/images/cuda_oom_screenshot.png",
    "log_snippet": "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
    "gold_answer": "The error occurs because your batch size is too large for GPU memory. ...",
    "gold_source_urls": ["https://pytorch.org/docs/stable/notes/cuda.html"],
    "gold_chunk_ids": [],
    "category": "cuda_memory",
    "difficulty": "easy"
  }
]
```

## Important: do not auto-generate the benchmark

It is tempting to use an LLM to generate question-answer pairs automatically.
This produces examples that are too easy and not credible — the model has memorized
the documentation and will answer correctly without retrieval. At least 60% of examples
should come from real engineering errors you encountered or reproduced yourself.
