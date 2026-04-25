import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a technical troubleshooting assistant.

Rules:
- Answer only using the documentation excerpts provided.
- Cite sources using [Source N].
- If the context is insufficient, say what is missing.
- Be practical and concise.
- Do not invent parameters, versions, or facts.
"""


def _build_context_block(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "Unknown")
        section = chunk.get("section", title)
        url = chunk.get("source_url", "")
        text = chunk.get("text", "")
        lines.append(
            f"[Source {i}] {title} — {section}\n"
            f"URL: {url}\n"
            f"{text}\n"
        )
    return "\n---\n".join(lines)


def _build_user_prompt(query: str, context_block: str) -> str:
    return (
        f"DOCUMENTATION EXCERPTS:\n\n{context_block}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"Give a grounded troubleshooting answer with citations."
    )


@dataclass
class GeneratorOutput:
    answer: str
    sources: list[dict] = field(default_factory=list)
    model_name: str = ""
    prompt_tokens: int = 0
    generation_time_s: float = 0.0


class _LocalModelBackend:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        logger.info(f"Loading local generator: {self.model_name}")
        self._pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
        self._load()

        prompt = (
            f"<|system|>\n{system_prompt}\n"
            f"<|user|>\n{user_prompt}\n"
            f"<|assistant|>\n"
        )

        out = self._pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            return_full_text=False,
        )
        return out[0]["generated_text"].strip()


class _OpenAIBackend:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Use local backend or export the key.")

        from openai import OpenAI
        client = OpenAI(api_key=api_key)
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


class _AnthropicBackend:
    def __init__(self, model_name: str = "claude-3-haiku-20240307"):
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Use local backend or export the key.")

        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text


class AnswerGenerator:
    def __init__(
        self,
        backend: Literal["local", "openai", "anthropic"] = "local",
        model_name: Optional[str] = None,
    ):
        if backend == "local":
            model = model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self._backend = _LocalModelBackend(model)
        elif backend == "openai":
            model = model_name or "gpt-4o-mini"
            self._backend = _OpenAIBackend(model)
        elif backend == "anthropic":
            model = model_name or "claude-3-haiku-20240307"
            self._backend = _AnthropicBackend(model)
        else:
            raise ValueError("backend must be local, openai, or anthropic")

        self.backend_name = backend
        self.model_name = model

    def generate(self, query: str, chunks: list[dict], max_new_tokens: int = 512) -> GeneratorOutput:
        if not chunks:
            return GeneratorOutput(
                answer="No relevant documentation was retrieved.",
                model_name=self.model_name,
            )

        context_block = _build_context_block(chunks)
        user_prompt = _build_user_prompt(query, context_block)
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
