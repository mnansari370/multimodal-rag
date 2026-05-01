"""
Adaptive query reformulator.

The ablation results showed that blindly appending logs/VLP text can hurt
retrieval when the user's question is already explicit. This module therefore
uses an adaptive strategy:

1. Explicit query:
   - keep the original query mostly unchanged.
   - avoids noisy logs/VLP hurting MRR.

2. Vague query:
   - inject VLP screenshot cues and concise error/log lines.
   - this is where multimodal parsing gives the largest gain.

3. Moderately specific query:
   - add only a small amount of high-signal context.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from ..multimodal.vlp import VLPOutput


logger = logging.getLogger(__name__)


@dataclass
class ReformulationResult:
    original_query: str
    reformulated_query: str
    vlp_context: str
    was_reformulated: bool
    strategy: str = "query_only"


ERROR_KEYWORDS = [
    "runtimeerror", "valueerror", "typeerror", "keyerror", "cuda", "oom",
    "nan", "error", "exception", "traceback", "failed", "nccl",
    "batchnorm", "groupnorm", "gradscaler", "autocast", "dataloader",
    "torch.compile", "checkpoint", "optimizer", "scheduler", "dtype",
    "float16", "bfloat16", "fp16", "bf16", "memory", "out of memory",
]


VAGUE_PATTERNS = [
    r"\bwhy is this happening\b",
    r"\bwhat is wrong\b",
    r"\bhow do i fix this\b",
    r"\bwhy does this fail\b",
    r"\bwhat does this mean\b",
    r"\bhelp me debug\b",
    r"\bfix this\b",
    r"\bdebug this\b",
]


def is_vague_query(query: str) -> bool:
    """Return True if the user query is too vague to retrieve well alone."""
    q = query.lower().strip()
    return any(re.search(p, q) for p in VAGUE_PATTERNS) or len(q.split()) <= 5


def is_specific_query(query: str) -> bool:
    """Return True if the query already contains useful technical terms."""
    q = query.lower()
    return any(k in q for k in ERROR_KEYWORDS)


def _extract_error_lines(log_text: str, max_lines: int = 2) -> list[str]:
    """
    Pull high-signal error lines from a log/config snippet.

    We avoid appending the whole log because the ablation showed that long
    logs can introduce retrieval noise.
    """
    if not log_text:
        return []

    lines = []
    for line in log_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(k in stripped.lower() for k in ERROR_KEYWORDS):
            lines.append(stripped)

    return lines[:max_lines]


def _vlp_to_query_text(vlp_output: Optional[VLPOutput], max_keywords: int = 8) -> str:
    """
    Convert VLPOutput into concise retrieval text.

    We include exact error message and selected keywords, but avoid dumping
    long raw descriptions.
    """
    if vlp_output is None:
        return ""

    parts = []

    if vlp_output.error_message:
        parts.append(vlp_output.error_message)

    if vlp_output.software_components:
        parts.append(" ".join(vlp_output.software_components[:5]))

    if vlp_output.keywords:
        parts.append(" ".join(vlp_output.keywords[:max_keywords]))

    return " ".join(parts).strip()


def reformulate_query(
    original_query: str,
    vlp_output: Optional[VLPOutput] = None,
    log_snippet: str = "",
) -> ReformulationResult:
    """
    Adaptive reformulation used by the final pipeline.

    Strategy:
    - If no VLP/log is present: query_only.
    - If query is explicit: keep original query to avoid noise.
    - If query is vague: add VLP + concise log lines.
    - Otherwise: add only concise error lines or concise VLP text.
    """
    original_query = original_query.strip()

    if vlp_output is None and not log_snippet:
        return ReformulationResult(
            original_query=original_query,
            reformulated_query=original_query,
            vlp_context="",
            was_reformulated=False,
            strategy="query_only",
        )

    query_is_vague = is_vague_query(original_query)
    query_is_specific = is_specific_query(original_query)

    vlp_text = _vlp_to_query_text(vlp_output)
    error_lines = _extract_error_lines(log_snippet, max_lines=2)

    # Case 1: explicit query already contains the useful technical terms.
    # Best retrieval ablation result came from avoiding noisy concatenation.
    if query_is_specific and not query_is_vague:
        return ReformulationResult(
            original_query=original_query,
            reformulated_query=original_query,
            vlp_context="",
            was_reformulated=False,
            strategy="explicit_query_only",
        )

    # Case 2: vague query. This is where VLP is critical.
    if query_is_vague:
        context_parts = []
        if vlp_text:
            context_parts.append(vlp_text)
        if error_lines:
            context_parts.extend(error_lines)

        vlp_context = " ".join(context_parts).strip()
        reformulated = f"{vlp_context} {original_query}".strip()

        return ReformulationResult(
            original_query=original_query,
            reformulated_query=reformulated,
            vlp_context=vlp_context,
            was_reformulated=bool(vlp_context),
            strategy="vague_query_with_context",
        )

    # Case 3: moderately specific query. Add concise context only.
    context_parts = []
    if error_lines:
        context_parts.extend(error_lines[:1])
    elif vlp_text:
        context_parts.append(vlp_text)

    vlp_context = " ".join(context_parts).strip()
    reformulated = f"{vlp_context} {original_query}".strip()

    return ReformulationResult(
        original_query=original_query,
        reformulated_query=reformulated,
        vlp_context=vlp_context,
        was_reformulated=bool(vlp_context),
        strategy="moderate_query_with_concise_context",
    )
