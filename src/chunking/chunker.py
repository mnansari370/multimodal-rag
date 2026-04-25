"""
Two chunking strategies for documentation pages:

1. Heading-based chunking  — splits on section boundaries. Each heading
   and everything below it (until the next same-level heading) becomes
   one chunk. Code blocks stay attached to the explanation paragraph
   that precedes them.

2. Fixed-size chunking with overlap — classic sliding window over the
   full page text, useful as a comparison baseline.

The project report specifically says to benchmark both strategies using
Recall@10 before committing to one. This module makes that easy.
"""

import json
import uuid
import logging
from pathlib import Path
from typing import Literal
from tqdm import tqdm


logger = logging.getLogger(__name__)

MAX_CHUNK_TOKENS = 400       # hard cap; we approximate 1 token ≈ 4 characters
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * 4
OVERLAP_CHARS = int(MAX_CHUNK_CHARS * 0.2)  # 20% overlap for fixed-size strategy


def _approx_tokens(text: str) -> int:
    return len(text) // 4


def _make_chunk(
    text: str,
    source_url: str,
    title: str,
    section: str,
    chunk_idx: int,
    strategy: str,
) -> dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "source_url": source_url,
        "title": title,
        "section": section,
        "text": text.strip(),
        "approx_tokens": _approx_tokens(text),
        "strategy": strategy,
        "chunk_idx": chunk_idx,
    }


# ─── Strategy 1: Heading-based ────────────────────────────────────────────────

def _split_by_headings(page: dict) -> list[dict]:
    """
    Build chunks by grouping content under each section heading.

    We walk the page's headings list and assign paragraphs/code blocks
    to the section they fall under. If a section is longer than
    MAX_CHUNK_CHARS we break it further at paragraph boundaries.
    """
    chunks = []
    headings = page.get("section_headings", [])
    paragraphs = page.get("paragraphs", [])
    code_blocks = page.get("code_blocks", [])
    notes = page.get("notes", [])

    # Flatten everything into a linear list of (type, text) pairs.
    # We use the heading list to mark section starts.
    # Paragraphs, code, and notes are interleaved as they appear on the page.
    #
    # Since we don't have line-level DOM ordering in our JSON, we approximate:
    # put paragraphs first, then code blocks (with their context), then notes.
    content_items = []

    for h in headings:
        content_items.append(("heading", h["level"], h["text"]))

    for p in paragraphs:
        content_items.append(("para", None, p))

    for cb in code_blocks:
        context_line = f"{cb['context']}\n" if cb["context"] else ""
        content_items.append(("code", None, f"{context_line}```\n{cb['code']}\n```"))

    for n in notes:
        content_items.append(("note", None, f"[{n['type'].upper()}] {n['text']}"))

    # Now reassemble into heading-delimited sections.
    # Each heading starts a new section; everything until the next heading
    # of the same or higher level belongs to that section.
    current_section = page["title"]
    current_buffer = []

    def flush(section_name, buffer, chunk_idx_start):
        if not buffer:
            return [], chunk_idx_start
        text = "\n\n".join(buffer)
        produced = []
        # Split long sections at paragraph boundaries
        if len(text) > MAX_CHUNK_CHARS:
            sub_chunks = _split_long_text(text)
            for i, sub in enumerate(sub_chunks):
                produced.append(
                    _make_chunk(sub, page["url"], page["title"], section_name,
                                chunk_idx_start + i, "heading")
                )
            return produced, chunk_idx_start + len(sub_chunks)
        else:
            return [_make_chunk(text, page["url"], page["title"], section_name,
                                chunk_idx_start, "heading")], chunk_idx_start + 1

    chunk_idx = 0
    for item_type, level, text in content_items:
        if item_type == "heading":
            new_chunks, chunk_idx = flush(current_section, current_buffer, chunk_idx)
            chunks.extend(new_chunks)
            current_section = text
            current_buffer = []
        else:
            current_buffer.append(text)

    # Flush the final section
    new_chunks, chunk_idx = flush(current_section, current_buffer, chunk_idx)
    chunks.extend(new_chunks)

    return chunks


def _split_long_text(text: str) -> list[str]:
    """Split text at paragraph boundaries when it exceeds MAX_CHUNK_CHARS."""
    paragraphs = text.split("\n\n")
    result = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > MAX_CHUNK_CHARS and current:
            result.append("\n\n".join(current))
            # Carry overlap: last paragraph continues into next chunk
            current = [current[-1]] if current else []
            current_len = len(current[0]) if current else 0
        current.append(para)
        current_len += para_len

    if current:
        result.append("\n\n".join(current))

    return result if result else [text[:MAX_CHUNK_CHARS]]


# ─── Strategy 2: Fixed-size with overlap ──────────────────────────────────────

def _split_fixed_size(page: dict) -> list[dict]:
    """
    Classic sliding-window chunking over the full page text.

    All text is concatenated, then cut into windows of MAX_CHUNK_CHARS
    with OVERLAP_CHARS overlap between consecutive windows.
    """
    all_text_parts = []
    all_text_parts.append(page["title"])
    all_text_parts.extend(page.get("paragraphs", []))
    for cb in page.get("code_blocks", []):
        if cb.get("context"):
            all_text_parts.append(cb["context"])
        all_text_parts.append(f"```\n{cb['code']}\n```")
    for n in page.get("notes", []):
        all_text_parts.append(f"[{n['type'].upper()}] {n['text']}")

    full_text = "\n\n".join(all_text_parts)

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(full_text):
        end = min(start + MAX_CHUNK_CHARS, len(full_text))
        window = full_text[start:end]

        # Try to end at a sentence boundary rather than mid-word
        if end < len(full_text):
            last_period = window.rfind(". ")
            if last_period > MAX_CHUNK_CHARS // 2:
                window = window[: last_period + 1]
                end = start + last_period + 1

        chunks.append(
            _make_chunk(window, page["url"], page["title"], page["title"],
                        chunk_idx, "fixed")
        )
        chunk_idx += 1
        # Advance by window size minus overlap
        start += len(window) - OVERLAP_CHARS

    return chunks


# ─── Public API ───────────────────────────────────────────────────────────────

def chunk_page(
    page: dict,
    strategy: Literal["heading", "fixed"] = "heading",
) -> list[dict]:
    """
    Chunk a single cleaned page dict.

    Args:
        page: Output from the cleaner (has url, title, paragraphs, etc.)
        strategy: 'heading' or 'fixed'

    Returns:
        List of chunk dicts, each with chunk_id, source_url, text, etc.
    """
    if strategy == "heading":
        return _split_by_headings(page)
    elif strategy == "fixed":
        return _split_fixed_size(page)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def chunk_all(
    processed_dir: str = "data/processed",
    output_file: str = "data/processed/chunks.jsonl",
    strategy: Literal["heading", "fixed"] = "heading",
) -> list[dict]:
    """
    Chunk all cleaned pages and write results to a JSONL file.

    Args:
        processed_dir: Directory with cleaned JSON files.
        output_file: Where to write the chunks (one JSON per line).
        strategy: Chunking strategy to use.

    Returns:
        All chunks as a list of dicts.
    """
    processed_path = Path(processed_dir)
    files = sorted(processed_path.glob("*.json"))
    logger.info(f"Chunking {len(files)} pages with strategy='{strategy}'")

    all_chunks = []
    for f in tqdm(files, desc=f"Chunking ({strategy})"):
        with open(f, "r", encoding="utf-8") as fh:
            page = json.load(fh)
        chunks = chunk_page(page, strategy)
        all_chunks.extend(chunks)

    # Write JSONL
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(all_chunks)} chunks to {output_file}")
    return all_chunks


def load_chunks(chunks_file: str = "data/processed/chunks.jsonl") -> list[dict]:
    """Load chunks from a JSONL file."""
    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk documentation pages")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-file", default="data/processed/chunks.jsonl")
    parser.add_argument("--strategy", choices=["heading", "fixed"], default="heading")
    args = parser.parse_args()

    chunks = chunk_all(args.processed_dir, args.output_file, args.strategy)
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Avg tokens/chunk: {sum(c['approx_tokens'] for c in chunks) // max(len(chunks), 1)}")
