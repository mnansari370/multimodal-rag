"""
End-to-end pipeline for the Multimodal RAG Troubleshooting System.

This is the main entry point that wires together all modules:

  [Input] → [VLP] → [Query Reformulator] → [Hybrid Retriever]
          → [Cross-encoder Reranker] → [Context Selector] → [Answer Generator]

The pipeline class is designed to be instantiated once (so models load
once) and then called repeatedly for different queries.
"""

import time
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
    """Configuration for the full pipeline. Mirrors configs/pipeline.yaml."""

    # Retrieval settings
    retriever_type: Literal["bm25", "dense", "hybrid"] = "hybrid"
    retrieval_top_k: int = 50          # candidates to pull before reranking
    reranker_top_k: int = 20          # candidates to keep after reranking

    # Context pruning settings
    pruning_strategy: Literal["threshold", "diversity", "coverage"] = "threshold"
    pruning_threshold: float = 0.0
    context_top_k: int = 5
    mmr_lambda: float = 0.7

    # Generation settings
    generator_backend: Literal["local", "openai", "anthropic"] = "openai"
    generator_model: str = "gpt-4o-mini"
    max_new_tokens: int = 512

    # VLP settings
    use_vlp: bool = True
    vlp_backend: Literal["internvl2", "llava"] = "internvl2"
    vlp_model: Optional[str] = None

    # Index paths
    bm25_path: str = "data/embeddings/bm25.pkl"
    faiss_path: str = "data/embeddings/dense.faiss"
    chunks_path: str = "data/embeddings/chunks.jsonl"


@dataclass
class PipelineOutput:
    """Full output from one pipeline run."""
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
    The full pipeline — instantiate once, query many times.

    Usage:
        pipeline = MultimodalRAGPipeline(config)
        pipeline.load_indexes()
        result = pipeline.run(
            question="Why is my training crashing?",
            image=screenshot_image,
            log_snippet="RuntimeError: CUDA out of memory",
        )
        print(result.answer)
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

        # Initialize components (models load lazily on first use)
        self.vlp = None
        if self.config.use_vlp:
            self.vlp = VisionLanguageParser(
                backend=self.config.vlp_backend,
                model_name=self.config.vlp_model,
            )

        self.retriever = self._build_retriever()
        self.reranker = Reranker()
        self.generator = AnswerGenerator(
            backend=self.config.generator_backend,
            model_name=self.config.generator_model,
        )

        self._indexes_loaded = False

    def _build_retriever(self):
        if self.config.retriever_type == "bm25":
            return BM25Retriever()
        elif self.config.retriever_type == "dense":
            return DenseRetriever()
        else:
            return HybridRetriever()

    def load_indexes(self):
        """Load pre-built retrieval indexes from disk."""
        logger.info("Loading retrieval indexes...")
        cfg = self.config

        if self.config.retriever_type == "bm25":
            self.retriever.load(cfg.bm25_path)
        elif self.config.retriever_type == "dense":
            self.retriever.load(cfg.faiss_path, cfg.chunks_path)
        else:
            self.retriever.load(cfg.bm25_path, cfg.faiss_path, cfg.chunks_path)

        self._indexes_loaded = True
        logger.info("Indexes loaded successfully.")

    def build_indexes(self, chunks: list[dict]):
        """Build retrieval indexes from a list of chunks (for initial setup)."""
        logger.info(f"Building indexes over {len(chunks)} chunks...")
        self.retriever.build(chunks)
        self._indexes_loaded = True

    def save_indexes(self):
        """Save built indexes to disk."""
        cfg = self.config
        if self.config.retriever_type == "bm25":
            self.retriever.save(cfg.bm25_path)
        elif self.config.retriever_type == "dense":
            self.retriever.save(cfg.faiss_path, cfg.chunks_path)
        else:
            self.retriever.save(cfg.bm25_path, cfg.faiss_path, cfg.chunks_path)
        logger.info("Indexes saved.")

    def run(
        self,
        question: str,
        image: Optional[Image.Image] = None,
        image_path: Optional[str] = None,
        log_snippet: str = "",
    ) -> PipelineOutput:
        """
        Run the full pipeline on a user query.

        Args:
            question: The user's typed question.
            image: Optional PIL Image (screenshot or diagram).
            image_path: Alternative to passing image directly.
            log_snippet: Optional log or config text pasted by user.

        Returns:
            PipelineOutput with the answer and all intermediate results.
        """
        if not self._indexes_loaded:
            raise RuntimeError("Indexes not loaded. Call load_indexes() or build_indexes() first.")

        latency = LatencyProfile()
        cfg = self.config

        # ── Module 2: Vision-Language Parser ────────────────────────────────
        vlp_output = None
        if self.vlp and (image is not None or image_path is not None):
            with Timer() as t:
                if image_path and image is None:
                    image = Image.open(image_path).convert("RGB")
                vlp_output = self.vlp.parse(image=image)
            latency.vlp_time_s = t.elapsed
            logger.info(f"VLP: category={vlp_output.visual_category}, "
                        f"error='{vlp_output.error_message[:60]}'")

        # ── Module 3: Query Reformulator ─────────────────────────────────────
        with Timer() as t:
            reformulation = reformulate_query(question, vlp_output, log_snippet)
        latency.reformulation_time_s = t.elapsed

        effective_query = reformulation.reformulated_query

        # ── Module 4: Retriever ───────────────────────────────────────────────
        with Timer() as t:
            retrieved = self.retriever.search(effective_query, top_k=cfg.retrieval_top_k)
        latency.retrieval_time_s = t.elapsed
        logger.info(f"Retrieved {len(retrieved)} candidates")

        # ── Module 5: Cross-encoder Reranker ──────────────────────────────────
        with Timer() as t:
            reranked = self.reranker.rerank(effective_query, retrieved, top_k=cfg.reranker_top_k)
        latency.reranking_time_s = t.elapsed
        logger.info(f"Reranked → {len(reranked)} candidates")

        # ── Module 6: Context Selector ────────────────────────────────────────
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
        logger.info(
            f"Context pruning ({cfg.pruning_strategy}): "
            f"{pruning_stats['original_tokens']} → {pruning_stats['pruned_tokens']} tokens "
            f"({pruning_stats['token_reduction_pct']}% reduction)"
        )

        # ── Module 7: Answer Generator ────────────────────────────────────────
        with Timer() as t:
            gen_output = self.generator.generate(
                query=effective_query,
                chunks=selected,
                max_new_tokens=cfg.max_new_tokens,
            )
        latency.generation_time_s = t.elapsed

        logger.info(
            f"Answer generated in {latency.generation_time_s:.2f}s | "
            f"Total latency: {latency.total_time_s:.2f}s"
        )

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
