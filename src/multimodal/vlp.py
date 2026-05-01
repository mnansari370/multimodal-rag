"""
Vision-Language Parser — converts a screenshot into structured retrieval cues.

When a user uploads a screenshot, we run a VLM on it once and extract the
error message, visual category, relevant software components, and keywords.
That structured text is then injected into the retrieval query by the
reformulator, which measurably improves retrieval for vague queries.

Supported backends: anthropic (Claude Haiku), openai (GPT-4o-mini),
internvl2 (local GPU, no API cost).
"""

import os
import re
import io
import json
import base64
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from PIL import Image


logger = logging.getLogger(__name__)


@dataclass
class VLPOutput:
    """Structured result from the vision-language parser."""
    error_message: str = ""
    visual_category: str = ""
    software_components: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    raw_description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_query_string(self) -> str:
        """Flatten all fields into a single retrieval string."""
        parts = []
        if self.error_message:
            parts.append(self.error_message)
        if self.visual_category:
            parts.append(self.visual_category)
        if self.software_components:
            parts.append(" ".join(self.software_components))
        if self.keywords:
            parts.append(" ".join(self.keywords))
        return " ".join(parts)


# Returning JSON makes parsing reliable across all backends and avoids
# brittle regex on free-form prose from different model families.
VLP_PROMPT = """Analyze this technical screenshot carefully. It is from a PyTorch
troubleshooting context.

Return ONLY valid JSON with exactly these four fields:
{
  "error_message": "the main visible error, warning, or issue — copy exact text if present, else empty string",
  "visual_category": "one of: stack_trace | config_file | terminal_output | architecture_diagram | training_plot | notebook_error | other",
  "software_components": ["list of frameworks/libraries/tools visible, e.g. PyTorch, CUDA, DataLoader, AMP, DDP"],
  "keywords": ["5 to 12 precise retrieval terms drawn from the screenshot — error names, function names, parameter names, tensor shapes"]
}

Be specific. Include exact error strings, tensor shapes, and parameter names when you can read them.
Do not include any text outside the JSON object.
"""


def _parse_vlp_json(raw: str) -> dict:
    """Pull the JSON object out of raw VLM output, handling markdown fences."""
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        logger.warning("VLP JSON parse failed on: %s", raw[:200])
        return {}


def _image_to_base64_png(image: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class _ClaudeVisionBackend:
    """Claude Haiku via Anthropic API. Fast and cheap for structured extraction."""

    def __init__(self, model_name: str = "claude-haiku-4-5-20251001"):
        self.model_name = model_name

    def describe(self, image: Image.Image) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")

        import anthropic

        b64_data = _image_to_base64_png(image)
        client = anthropic.Anthropic(api_key=api_key)

        msg = client.messages.create(
            model=self.model_name,
            max_tokens=400,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_data,
                            },
                        },
                        {"type": "text", "text": VLP_PROMPT},
                    ],
                }
            ],
        )
        return msg.content[0].text.strip()


class _OpenAIVisionBackend:
    """GPT-4o-mini via OpenAI API."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

    def describe(self, image: Image.Image) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        from openai import OpenAI

        b64_data = _image_to_base64_png(image)
        data_url = f"data:image/png;base64,{b64_data}"

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLP_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()


class _InternVL2Backend:
    """
    InternVL2-2B or 7B running locally on GPU.

    Use this when you want zero API cost and have a GPU available.
    The 2B model is fast; 7B gives better extraction on complex screenshots.
    """

    def __init__(self, model_name: str = "OpenGVLab/InternVL2-2B"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModel

        logger.info("Loading VLM: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        ).eval()

    def describe(self, image: Image.Image) -> str:
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        self._load()

        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        pixel_values = transform(image.convert("RGB")).unsqueeze(0)
        device = next(self._model.parameters()).device
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pixel_values = pixel_values.to(device=device, dtype=dtype)

        return self._model.chat(
            self._tokenizer,
            pixel_values,
            VLP_PROMPT,
            {"max_new_tokens": 400, "do_sample": False},
        )


class VisionLanguageParser:
    """Takes a screenshot and returns structured VLPOutput for the reformulator."""

    BACKENDS = {
        "anthropic": _ClaudeVisionBackend,
        "openai": _OpenAIVisionBackend,
        "internvl2": _InternVL2Backend,
    }

    def __init__(self, backend: str = "anthropic", model_name: Optional[str] = None):
        if backend not in self.BACKENDS:
            raise ValueError(
                f"Unknown VLP backend '{backend}'. "
                f"Choose from: {list(self.BACKENDS)}"
            )
        cls = self.BACKENDS[backend]
        self._backend = cls(model_name) if model_name else cls()
        logger.info("VLP initialized with backend=%s", backend)

    def parse(
        self,
        image: Optional[Image.Image] = None,
        image_path: Optional[str] = None,
    ) -> VLPOutput:
        """Parse a screenshot and return structured visual cues."""
        if image is None and image_path is None:
            return VLPOutput()

        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")

        logger.debug("Running VLP on image (size=%s)", image.size)
        raw = self._backend.describe(image)
        parsed = _parse_vlp_json(raw)

        return VLPOutput(
            error_message=parsed.get("error_message", ""),
            visual_category=parsed.get("visual_category", "other"),
            software_components=parsed.get("software_components", []),
            keywords=parsed.get("keywords", []),
            raw_description=raw,
        )
