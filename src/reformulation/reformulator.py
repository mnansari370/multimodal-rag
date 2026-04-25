"""
Query reformulator.

After the VLP produces structured cues from a screenshot, this module
rewrites the user's original query by injecting those cues. This
measurably improves retrieval because the reformulated query contains
the actual error terms rather than a vague natural language question.

Example:
  Original: "Why is this happening?"
  VLP output: error="RuntimeError: CUDA out of memory", keywords=["batch_size", "64"]
  Reformulated: "RuntimeError CUDA out of memory batch_size 64 why is this happening"

This step also enables a clean ablation study:
  - Retrieval with original query
  - Retrieval with reformulated query
"""

import logging
from dataclasses import dataclass
from typing import Optional

from ..multimodal.vlp import VLPOutput


logger = logging.getLogger(__name__)


@dataclass
class ReformulationResult:
    original_query: str
    reformulated_query: str
    vlp_context: str          # the VLP string that was injected
    was_reformulated: bool    # False if no image was provided


def reformulate_query(
    original_query: str,
    vlp_output: Optional[VLPOutput] = None,
    log_snippet: str = "",
) -> ReformulationResult:
    """
    Combine the original query with VLP cues and any log/config text.

    Strategy:
    1. If we have a VLP output, prepend its error message and keywords.
    2. If a log snippet is provided, extract the first line that contains
       an error or exception and prepend that too.
    3. Append the original query at the end so its intent is preserved.

    Args:
        original_query: The user's typed question.
        vlp_output: Structured output from the VisionLanguageParser.
        log_snippet: Optional raw log or config text pasted by the user.

    Returns:
        ReformulationResult with both the original and reformulated query.
    """
    if vlp_output is None and not log_snippet:
        return ReformulationResult(
            original_query=original_query,
            reformulated_query=original_query,
            vlp_context="",
            was_reformulated=False,
        )

    parts = []

    # Add VLP-derived context
    if vlp_output:
        if vlp_output.error_message:
            parts.append(vlp_output.error_message)
        if vlp_output.software_components:
            parts.append(" ".join(vlp_output.software_components))
        if vlp_output.keywords:
            parts.append(" ".join(vlp_output.keywords))

    # Add relevant lines from log/config pasted by user
    if log_snippet:
        error_lines = _extract_error_lines(log_snippet)
        if error_lines:
            parts.extend(error_lines[:3])  # top 3 error lines

    vlp_context = " ".join(parts)

    # Combine: VLP context first (for retrieval weight), original query last
    reformulated = f"{vlp_context} {original_query}".strip()

    logger.debug(f"Reformulated: '{original_query}' → '{reformulated[:100]}...'")

    return ReformulationResult(
        original_query=original_query,
        reformulated_query=reformulated,
        vlp_context=vlp_context,
        was_reformulated=True,
    )


def _extract_error_lines(log_text: str) -> list[str]:
    """
    Pull out lines from a log or stack trace that contain error indicators.

    We look for common Python exception patterns, CUDA errors, and
    lines with 'Error', 'Exception', 'FAILED', 'Traceback'.
    """
    error_keywords = [
        "Error", "Exception", "FAILED", "Traceback", "fatal",
        "RuntimeError", "ValueError", "TypeError", "KeyError",
        "CUDA", "OOM", "killed", "Segfault",
    ]
    lines = log_text.splitlines()
    error_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(kw.lower() in stripped.lower() for kw in error_keywords):
            error_lines.append(stripped)

    return error_lines
