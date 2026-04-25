# Multimodal RAG Troubleshooter — Current Status

## Corpus
- Source: PyTorch documentation
- Raw pages: 2488
- Cleaned pages: 1691
- Heading chunks: 6047
- Fixed chunks: 6064

## Benchmark
- Total examples: 70
- Inputs: question + optional logs/configs + optional screenshots
- Gold answers: 70
- Gold source URLs: 70
- Screenshot examples: 2

## Completed Experiments
- BM25 vs Dense vs Hybrid vs Hybrid+Reranker
- Heading vs Fixed chunking
- Context pruning ablation
- Failure analysis candidate extraction
- End-to-end generation evaluation running on SLURM

## Current Limitation
- VLP/InternVL2 disabled during generation eval because current transformers version has compatibility issues with InternVL2 remote code.
- Current generation eval is text/log-based.
- More screenshot examples are needed for stronger multimodal evaluation.

## Core Innovation
Context-efficient grounding using threshold, diversity, and MMR-style coverage pruning.
