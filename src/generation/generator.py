"""
Answer generator — takes a query and selected context chunks, produces
a grounded answer with [Source N] citations.

The Anthropic API returns accurate token counts (msg.usage.input_tokens),
which is what we use for efficiency tracking.
"""

import os
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_time_s: float = 0.0


class AnswerGenerator:
    """Wraps the Anthropic Claude API to produce cited answers from context."""

    def __init__(
        self,
        backend: str = "anthropic",
        model_name: str = "claude-haiku-4-5-20251001",
    ):
        self.backend = backend
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
        """Generate a cited answer from the selected context chunks."""
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
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")

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

        return GeneratorOutput(
            answer=msg.content[0].text,
            sources=chunks,
            model_name=self.model_name,
            prompt_tokens=msg.usage.input_tokens,
            completion_tokens=msg.usage.output_tokens,
            generation_time_s=round(elapsed, 3),
        )
