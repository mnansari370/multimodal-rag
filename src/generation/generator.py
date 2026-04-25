"""
Answer generator with citation prompting.

Takes the selected evidence chunks and the user's query, then asks an LLM
to produce a grounded answer that cites its sources. The prompt format
is designed to minimize hallucination — the model is explicitly told to
only use information from the provided context.

Supports two backends:
  - Local: Llama-3-8B-Instruct or Mistral-7B-Instruct via HuggingFace
  - API: OpenAI-compatible endpoints (GPT-4o, GPT-4o-mini) or Anthropic Claude
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal


logger = logging.getLogger(__name__)


# ─── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a technical troubleshooting assistant. You answer questions \
about errors, configurations, and software issues based only on the documentation \
excerpts provided.

Rules:
- Every claim you make must be traceable to a specific excerpt below.
- After each key claim, cite the source using [Source N] notation.
- If the excerpts do not contain enough information, say so clearly.
- Do not make up facts, version numbers, or parameter names.
- Be direct and practical — the user wants a fix, not a lecture."""


def _build_context_block(chunks: list[dict]) -> str:
    """Format selected chunks into a numbered reference list."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", "Unknown")
        section = chunk.get("section", title)
        url = chunk.get("source_url", "")
        text = chunk["text"]
        lines.append(
            f"[Source {i}] {title} — {section}\n"
            f"URL: {url}\n"
            f"{text}\n"
        )
    return "\n---\n".join(lines)


def _build_user_prompt(query: str, context_block: str) -> str:
    return (
        f"DOCUMENTATION EXCERPTS:\n\n{context_block}\n\n"
        f"---\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer based only on the excerpts above. Cite sources using [Source N]."
    )


# ─── Answer dataclass ─────────────────────────────────────────────────────────

@dataclass
class GeneratorOutput:
    answer: str
    sources: list[dict] = field(default_factory=list)   # chunks used as context
    model_name: str = ""
    prompt_tokens: int = 0
    generation_time_s: float = 0.0


# ─── Local model backend ──────────────────────────────────────────────────────

class _LocalModelBackend:
    """Runs a HuggingFace Instruct model locally."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        logger.info(f"Loading local model: {self.model_name}")
        self._pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
        self._load()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        out = self._pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
        # HuggingFace pipeline returns the full conversation; extract the last assistant turn
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1]["content"]
        return generated


# ─── API backend (OpenAI-compatible) ─────────────────────────────────────────

class _OpenAIBackend:
    """Calls an OpenAI-compatible API (GPT-4o-mini, GPT-4o, etc.)."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content


# ─── Anthropic Claude backend ─────────────────────────────────────────────────

class _AnthropicBackend:
    """Calls Claude via the Anthropic API."""

    def __init__(self, model_name: str = "claude-haiku-4-5-20251001"):
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text


# ─── Public API ───────────────────────────────────────────────────────────────

class AnswerGenerator:
    """
    Generates grounded, cited answers from selected evidence chunks.

    Usage:
        gen = AnswerGenerator(backend="openai", model_name="gpt-4o-mini")
        output = gen.generate(query="Why is my training crashing?", chunks=selected_chunks)
        print(output.answer)
    """

    def __init__(
        self,
        backend: Literal["local", "openai", "anthropic"] = "openai",
        model_name: Optional[str] = None,
    ):
        if backend == "local":
            model = model_name or "meta-llama/Meta-Llama-3-8B-Instruct"
            self._backend = _LocalModelBackend(model)
        elif backend == "openai":
            model = model_name or "gpt-4o-mini"
            self._backend = _OpenAIBackend(model)
        elif backend == "anthropic":
            model = model_name or "claude-haiku-4-5-20251001"
            self._backend = _AnthropicBackend(model)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose: local | openai | anthropic")

        self.backend_name = backend
        self.model_name = model_name or self._backend.model_name

    def generate(
        self,
        query: str,
        chunks: list[dict],
        max_new_tokens: int = 512,
    ) -> GeneratorOutput:
        """
        Generate a cited answer from the provided chunks.

        Args:
            query: User's question (original or reformulated).
            chunks: Selected context chunks from the efficiency module.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            GeneratorOutput with the answer and metadata.
        """
        if not chunks:
            return GeneratorOutput(
                answer="No relevant documentation was found for this query. "
                       "Please try rephrasing or providing more context.",
                model_name=self.model_name,
            )

        context_block = _build_context_block(chunks)
        user_prompt = _build_user_prompt(query, context_block)

        # Rough token count estimate for logging
        prompt_tokens = len((SYSTEM_PROMPT + user_prompt).split())

        start = time.time()
        answer = self._backend.generate(SYSTEM_PROMPT, user_prompt, max_new_tokens)
        elapsed = time.time() - start

        return GeneratorOutput(
            answer=answer,
            sources=chunks,
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            generation_time_s=round(elapsed, 3),
        )
