"""
Answer generator (Module 7 of the pipeline).

Takes the user's (reformulated) query and the selected context chunks,
and produces a grounded answer with [Source N] citations.

The generator is intentionally kept as a simple API call. The intelligence
is in the retrieval, reranking, and context-selection stages that come
before it — by the time we reach generation, the context should already
contain the right information.

Using the Anthropic API with Claude gives us accurate token counts from
the API response itself (msg.usage.input_tokens), which is important for
the efficiency ablation study in Section 6.2.
"""

import os
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# The system prompt tells the model to behave as a RAG assistant:
# answer only from provided context, always cite sources, be concise.
SYSTEM_PROMPT = """You are a PyTorch technical troubleshooting assistant.
Your job is to diagnose and explain errors based on the provided documentation context.

Rules you must follow:
1. Answer ONLY using information from the provided context. Do not use outside knowledge.
2. Every factual claim must have a citation like [Source 1] or [Source 2].
3. If the provided context does not contain enough information to answer, say so clearly.
4. Be direct and practical — engineers need working fixes, not long preambles.

Structure your answer as:

**Root cause:**
- [what is causing the issue, with citation]

**Fix:**
- [concrete steps to resolve it, with citations]

**Why this works:**
- [brief explanation grounded in the docs]

**Sources used:**
- [Source N]: [URL]
"""


@dataclass
class GeneratorOutput:
    """Result from the answer generator."""
    answer: str
    sources: list = field(default_factory=list)
    model_name: str = ""
    prompt_tokens: int = 0       # input tokens as reported by the API
    completion_tokens: int = 0   # output tokens
    generation_time_s: float = 0.0


class AnswerGenerator:
    """
    Module 7: Answer generator.

    Wraps the Anthropic Claude API to produce cited answers from context.
    Token counts come from the API usage response, which is accurate for
    the efficiency ablation table.

    Usage:
        gen = AnswerGenerator()
        out = gen.generate(query="Why is training crashing?", chunks=reranked_chunks)
        print(out.answer)
    """

    def __init__(
        self,
        backend: str = "anthropic",
        model_name: str = "claude-haiku-4-5-20251001",
    ):
        self.backend = backend
        # Use Sonnet for higher answer quality, Haiku for speed/cost
        self.model_name = model_name

    def _build_context_block(self, chunks: list[dict]) -> str:
        """Format retrieved chunks as a numbered context block."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            url = chunk.get("source_url", "")
            title = chunk.get("title", "Unknown")
            section = chunk.get("section", "")
            text = chunk.get("text", "")

            header = f"[Source {i}] {title}"
            if section and section != title:
                header += f" — {section}"

            parts.append(f"{header}\nURL: {url}\n\n{text}")

        return "\n\n---\n\n".join(parts)

    def _build_user_prompt(self, query: str, chunks: list[dict]) -> str:
        """Combine context and query into the user-turn message."""
        context = self._build_context_block(chunks)
        return (
            f"Documentation context:\n\n{context}\n\n"
            f"---\n\n"
            f"Question: {query}\n\n"
            f"Write a grounded answer with [Source N] citations for every claim."
        )

    def generate(
        self,
        query: str,
        chunks: list[dict],
        max_new_tokens: int = 500,
    ) -> GeneratorOutput:
        """
        Generate a cited answer from the selected context chunks.

        Args:
            query: The (possibly reformulated) user query.
            chunks: Context chunks selected by the pruning stage.
            max_new_tokens: Max tokens for the generated answer.

        Returns:
            GeneratorOutput with the answer, token counts, and timing.
        """
        if not chunks:
            logger.warning("No context chunks provided to generator.")
            return GeneratorOutput(
                answer="No relevant documentation was retrieved for this query.",
                model_name=self.model_name,
            )

        user_prompt = self._build_user_prompt(query, chunks)

        if self.backend == "anthropic":
            return self._generate_anthropic(user_prompt, chunks, max_new_tokens)
        else:
            raise ValueError(
                f"Unknown generator backend: '{self.backend}'. "
                "Only 'anthropic' is supported."
            )

    def _generate_anthropic(
        self,
        user_prompt: str,
        chunks: list[dict],
        max_new_tokens: int,
    ) -> GeneratorOutput:
        """Call the Anthropic API and return a GeneratorOutput."""
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it before running generation."
            )

        client = anthropic.Anthropic(api_key=api_key)

        start = time.perf_counter()
        msg = client.messages.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        elapsed = time.perf_counter() - start

        answer = msg.content[0].text

        # Token counts come directly from the API response — more accurate
        # than any local approximation, and these are what we report in
        # the efficiency ablation table.
        return GeneratorOutput(
            answer=answer,
            sources=chunks,
            model_name=self.model_name,
            prompt_tokens=msg.usage.input_tokens,
            completion_tokens=msg.usage.output_tokens,
            generation_time_s=round(elapsed, 3),
        )
