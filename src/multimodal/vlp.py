"""
Vision-Language Parser (VLP).

When a user uploads a screenshot, this module uses a Vision-Language Model
to produce a structured text interpretation of the image. We treat the VLM
as a 'screenshot-to-text' preprocessing step rather than building a
multimodal retriever from scratch — this keeps the rest of the pipeline
text-based and makes ablation studies clean.

The output is a structured dict containing:
  - error_message: the main error text or warning visible in the image
  - visual_category: type of image (stack trace, plot, UI error, architecture diagram, config)
  - software_components: any framework/library names detected
  - keywords: retrieval-friendly terms extracted from the image

Supports two VLMs:
  - InternVL2-2B / InternVL2-8B (recommended for HPC — efficient, strong)
  - LLaVA-1.6-7B (fallback option)
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from PIL import Image


logger = logging.getLogger(__name__)


@dataclass
class VLPOutput:
    """Structured output from the Vision-Language Parser."""
    error_message: str = ""
    visual_category: str = ""          # stack_trace | config | plot | ui_error | diagram | other
    software_components: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    raw_description: str = ""          # full VLM output before parsing

    def to_dict(self) -> dict:
        return asdict(self)

    def to_query_string(self) -> str:
        """Flatten into a string suitable for query reformulation."""
        parts = []
        if self.error_message:
            parts.append(self.error_message)
        if self.software_components:
            parts.append(" ".join(self.software_components))
        if self.keywords:
            parts.append(" ".join(self.keywords))
        return " ".join(parts)


# ─── Prompt used to instruct the VLM ─────────────────────────────────────────

VLP_PROMPT = """You are analyzing a technical screenshot for a troubleshooting system.

Examine this image carefully and respond with a JSON object containing exactly these fields:

{
  "error_message": "the main error, warning, or exception message visible (empty string if none)",
  "visual_category": "one of: stack_trace | config_file | plot_or_graph | ui_error | architecture_diagram | terminal_output | other",
  "software_components": ["list of framework or library names you recognize, e.g. PyTorch, CUDA, Kubernetes"],
  "keywords": ["5 to 10 specific technical terms useful for searching documentation"]
}

Be specific. Include full error strings, version numbers, and parameter names if visible.
Return only the JSON object — no explanation text."""


def _parse_vlp_response(raw: str) -> dict:
    """Extract the JSON block from the VLM's response text."""
    # Try to find a JSON block — the model sometimes wraps it in markdown
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return {}
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return {}


# ─── Model backends ───────────────────────────────────────────────────────────

class _InternVL2Backend:
    """InternVL2 backend. Handles both 2B and 8B variants."""

    def __init__(self, model_name: str = "OpenGVLab/InternVL2-2B"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModel

        logger.info(f"Loading VLM: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def describe(self, image: Image.Image) -> str:
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        self._load()

        # InternVL2 expects a specific image transform
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        pixel_values = transform(image.convert("RGB")).unsqueeze(0)
        device = next(self._model.parameters()).device
        pixel_values = pixel_values.to(device, dtype=torch.float16)

        generation_config = {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 1.0,
        }

        response = self._model.chat(
            self._tokenizer,
            pixel_values,
            VLP_PROMPT,
            generation_config,
        )
        return response


class _LLaVABackend:
    """LLaVA-1.6 backend."""

    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf"):
        self.model_name = model_name
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        logger.info(f"Loading VLM: {self.model_name}")
        self._pipe = pipeline(
            "image-to-text",
            model=self.model_name,
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )

    def describe(self, image: Image.Image) -> str:
        self._load()
        prompt = f"USER: <image>\n{VLP_PROMPT}\nASSISTANT:"
        out = self._pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 512})
        return out[0]["generated_text"].split("ASSISTANT:")[-1].strip()


# ─── Public API ───────────────────────────────────────────────────────────────

class VisionLanguageParser:
    """
    Wraps a VLM and produces structured VLPOutput from screenshots.

    Usage:
        vlp = VisionLanguageParser(backend="internvl2")
        result = vlp.parse(image_path="screenshot.png")
        print(result.error_message)
        print(result.to_query_string())
    """

    BACKENDS = {
        "internvl2": _InternVL2Backend,
        "llava": _LLaVABackend,
    }

    def __init__(
        self,
        backend: str = "internvl2",
        model_name: Optional[str] = None,
    ):
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(self.BACKENDS)}")

        BackendClass = self.BACKENDS[backend]
        self._backend = BackendClass(model_name) if model_name else BackendClass()

    def parse(
        self,
        image: Image.Image | str | None = None,
        image_path: str | None = None,
    ) -> VLPOutput:
        """
        Parse an image and return structured VLPOutput.

        Pass either a PIL Image or a file path.
        Returns an empty VLPOutput if no image is provided.
        """
        if image is None and image_path is None:
            return VLPOutput()

        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")

        raw = self._backend.describe(image)
        parsed = _parse_vlp_response(raw)

        return VLPOutput(
            error_message=parsed.get("error_message", ""),
            visual_category=parsed.get("visual_category", "other"),
            software_components=parsed.get("software_components", []),
            keywords=parsed.get("keywords", []),
            raw_description=raw,
        )
