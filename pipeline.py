"""
Multimodal RAG Pipeline — the top-level orchestrator.

Wires together all seven modules described in the project report:
  1. Input processor (handled by this class's run() method)
  2. Vision-language parser (VisionLanguageParser)
  3. Query reformulator (reformulate_query)
  4. Retriever (BM25Retriever / DenseRetriever / HybridRetriever)
  5. Cross-encoder reranker (Reranker)
  6. Context selector (select_context) — the core innovation
  7. Answer generator (AnswerGenerator)

Usage:
    cfg = PipelineConfig(generator_model="claude-haiku-4-5-20251001")
    pipeline = MultimodalRAGPipeline(cfg)
    pipeline.load_indexes()
    result = pipeline.run(question="Why is training crashing?", image_path="error.png")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

from PIL import Image

from src.multimodal import VisionLanguageParser, VLPOutput
from src.reformulation import reformulate_query, ReformulationResult
from src.retrieval import HybridRetriever, BM25Retriever, DenseRetriever
from src.reranking import Reranker
from src.efficiency import select_context
from src.generation import AnswerGenerator, GeneratorOutput
from src.evaluation.efficiency_metrics import LatencyProfile, Timer


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    All knobs for a single pipeline run.

    Designed so you can swap components in and out cleanly for the
    ablation study — e.g. set use_reranker=False to measure the
    reranker's contribution, or change pruning_strategy to compare
    the three context-selection mechanisms.
    """

    # ── Retrieval ──────────────────────────────────────────────────────────
    retriever_type: Literal["bm25", "dense", "hybrid"] = "hybrid"
    retrieval_top_k: int = 50          # candidates to pull before reranking

    # ── Reranking ──────────────────────────────────────────────────────────
    use_reranker: bool = True
    reranker_top_k: int = 20           # top-k to keep after reranking

    # ── Context pruning (the core innovation) ──────────────────────────────
    pruning_strategy: Literal["threshold", "diversity", "coverage"] = "coverage"
    pruning_threshold: float = 0.0     # for threshold strategy
    context_top_k: int = 5             # final chunks sent to generator
    mmr_lambda: float = 0.7            # coverage MMR trade-off

    # ── Generation ─────────────────────────────────────────────────────────
    generator_backend: str = "anthropic"
    generator_model: str = "claude-haiku-4-5-20251001"
    max_new_tokens: int = 500

    # ── Vision-language parser ─────────────────────────────────────────────
    use_vlp: bool = True
    vlp_backend: Literal["anthropic", "openai", "internvl2"] = "anthropic"
    vlp_model: Optional[str] = None    # None = use backend's default

    # ── Index paths ────────────────────────────────────────────────────────
    bm25_path: str = "data/embeddings/bm25.pkl"
    faiss_path: str = "data/embeddings/dense.faiss"
    chunks_path: str = "data/embeddings/chunks.jsonl"


@dataclass
class PipelineOutput:
    """Everything produced by a single pipeline.run() call."""
    answer: str
    sources: list[dict] = field(default_factory=list)
    vlp_output: Optional[VLPOutput] = None
    reformulation: Optional[ReformulationResult] = None
    retrieved_chunks: list[dict] = field(default_factory=list)
    reranked_chunks: list[dict] = field(default_factory=list)
    selected_chunks: list[dict] = field(default_factory=list)
    pruning_stats: dict = field(default_factory=dict)
    latency: Optional[LatencyProfile] = None
    generator_output: Optional[GeneratorOutput] = None


class MultimodalRAGPipeline:
    """
    Orchestrates all pipeline modules end-to-end.

    The pipeline is stateful: you must call load_indexes() (or
    build_indexes()) before calling run(). This mirrors how a real
    production system works — indexes are loaded once at startup and
    then queries are served against them.
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

        # VLP is optional — only initialize if the config requests it
        self.vlp = None
        if self.config.use_vlp:
            self.vlp = VisionLanguageParser(
                backend=self.config.vlp_backend,
                model_name=self.config.vlp_model,
            )

        self.retriever = self._build_retriever()

        # The reranker is loaded lazily (model downloads on first use)
        self.reranker = Reranker() if self.config.use_reranker else None

        self.generator = AnswerGenerator(
            backend=self.config.generator_backend,
            model_name=self.config.generator_model,
        )

        self._indexes_loaded = False

    def _build_retriever(self):
        """Instantiate the configured retriever type."""
        if self.config.retriever_type == "bm25":
            return BM25Retriever()
        if self.config.retriever_type == "dense":
            return DenseRetriever()
        return HybridRetriever()

    def load_indexes(self):
        """Load pre-built retrieval indexes from disk."""
        cfg = self.config
        logger.info("Loading retrieval indexes from disk...")

        if cfg.retriever_type == "bm25":
            self.retriever.load(cfg.bm25_path)
        elif cfg.retriever_type == "dense":
            self.retriever.load(cfg.faiss_path, cfg.chunks_path)
        else:
            # Hybrid loads both
            self.retriever.load(cfg.bm25_path, cfg.faiss_path, cfg.chunks_path)

        self._indexes_loaded = True
        logger.info("Indexes loaded.")

    def build_indexes(self, chunks: list[dict]):
        """Build indexes in-memory from a list of chunk dicts (for scripting)."""
        self.retriever.build(chunks)
        self._indexes_loaded = True

    def save_indexes(self):
        """Save in-memory indexes to the configured paths."""
        cfg = self.config
        if cfg.retriever_type == "bm25":
            self.retriever.save(cfg.bm25_path)
        elif cfg.retriever_type == "dense":
            self.retriever.save(cfg.faiss_path, cfg.chunks_path)
        else:
            self.retriever.save(cfg.bm25_path, cfg.faiss_path, cfg.chunks_path)

    def retrieve_only(self, query: str, top_k: int = 20) -> list[dict]:
        """Run just the retrieval step. Used by the retrieval evaluation script."""
        if not self._indexes_loaded:
            raise RuntimeError("Indexes not loaded. Call load_indexes() first.")
        return self.retriever.search(query, top_k=top_k)

    def run(
        self,
        question: str,
        image: Optional[Image.Image] = None,
        image_path: Optional[str] = None,
        log_snippet: str = "",
    ) -> PipelineOutput:
        """
        Run the full pipeline end-to-end on a single query.

        Args:
            question: The user's natural-language question.
            image: A PIL Image of a screenshot (optional).
            image_path: Path to a screenshot file (alternative to image).
            log_snippet: Pasted log output or config text (optional).

        Returns:
            PipelineOutput with the answer, sources, and timing breakdown.
        """
        if not self._indexes_loaded:
            raise RuntimeError("Indexes not loaded. Call load_indexes() first.")

        cfg = self.config
        latency = LatencyProfile()

        # ── Module 2: Vision-language parser ──────────────────────────────
        vlp_output = None
        if self.vlp and (image is not None or image_path is not None):
            with Timer() as t:
                if image is None and image_path:
                    image = Image.open(image_path).convert("RGB")
                vlp_output = self.vlp.parse(image=image)
            latency.vlp_time_s = t.elapsed
            logger.debug("VLP: %s", vlp_output.error_message[:80])

        # ── Module 3: Query reformulator ───────────────────────────────────
        with Timer() as t:
            reformulation = reformulate_query(question, vlp_output, log_snippet)
        latency.reformulation_time_s = t.elapsed
        effective_query = reformulation.reformulated_query

        # ── Module 4: Retriever ────────────────────────────────────────────
        with Timer() as t:
            retrieved = self.retriever.search(effective_query, top_k=cfg.retrieval_top_k)
        latency.retrieval_time_s = t.elapsed
        logger.debug("Retrieved %d candidates", len(retrieved))

        # ── Module 5: Cross-encoder reranker ──────────────────────────────
        if self.reranker:
            with Timer() as t:
                reranked = self.reranker.rerank(
                    effective_query,
                    retrieved,
                    top_k=cfg.reranker_top_k,
                )
            latency.reranking_time_s = t.elapsed
        else:
            # No reranker — take top candidates as-is and add a dummy score
            reranked = retrieved[:cfg.reranker_top_k]
            for c in reranked:
                c.setdefault(
                    "reranker_score",
                    c.get("rrf_score", c.get("dense_score", c.get("bm25_score", 0.0))),
                )

        # ── Module 6: Context selector ────────────────────────────────────
        with Timer() as t:
            selected, pruning_stats = select_context(
                reranked,
                strategy=cfg.pruning_strategy,
                query=effective_query,
                threshold=cfg.pruning_threshold,
                top_k=cfg.context_top_k,
                lambda_param=cfg.mmr_lambda,
            )
        latency.pruning_time_s = t.elapsed
        logger.debug(
            "Context pruning (%s): %d → %d chunks",
            cfg.pruning_strategy, len(reranked), len(selected),
        )

        # ── Module 7: Answer generator ────────────────────────────────────
        with Timer() as t:
            gen_output = self.generator.generate(
                query=effective_query,
                chunks=selected,
                max_new_tokens=cfg.max_new_tokens,
            )
        latency.generation_time_s = t.elapsed

        return PipelineOutput(
            answer=gen_output.answer,
            sources=selected,
            vlp_output=vlp_output,
            reformulation=reformulation,
            retrieved_chunks=retrieved,
            reranked_chunks=reranked,
            selected_chunks=selected,
            pruning_stats=pruning_stats,
            latency=latency,
            generator_output=gen_output,
        )
