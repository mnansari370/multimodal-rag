"""
Clean and normalize raw documentation JSON files into a consistent format.

Takes the raw output from downloader.py and produces clean, deduplicated
records suitable for chunking. Removes boilerplate text, very short pages,
and pages that are index-only listings.
"""

import json
import re
import logging
from pathlib import Path
from tqdm import tqdm


logger = logging.getLogger(__name__)

# Strings that indicate a page is just navigation / index
BOILERPLATE_PATTERNS = [
    r"^Table of Contents$",
    r"^Search Page$",
    r"^Index$",
    r"^Module Index$",
    r"^Note$",
]

MIN_PARAGRAPH_CHARS = 50
MIN_PARAGRAPHS_PER_PAGE = 2


def clean_text(text: str) -> str:
    """Normalize whitespace and strip leftover HTML entities."""
    text = re.sub(r"\s+", " ", text)
    text = text.replace("¶", "").replace("¶", "")
    text = re.sub(r"&[a-z]+;", " ", text)
    return text.strip()


def is_boilerplate_title(title: str) -> bool:
    for pat in BOILERPLATE_PATTERNS:
        if re.match(pat, title.strip(), re.IGNORECASE):
            return True
    return False


def clean_page(raw: dict) -> dict | None:
    """
    Clean a single raw page dict.

    Returns None if the page should be discarded.
    """
    title = clean_text(raw.get("title", ""))

    if is_boilerplate_title(title):
        return None

    paragraphs = [clean_text(p) for p in raw.get("paragraphs", [])]
    paragraphs = [p for p in paragraphs if len(p) >= MIN_PARAGRAPH_CHARS]

    if len(paragraphs) < MIN_PARAGRAPHS_PER_PAGE:
        return None

    code_blocks = []
    for cb in raw.get("code_blocks", []):
        code = cb.get("code", "").strip()
        context = clean_text(cb.get("context", ""))
        if code:
            code_blocks.append({"code": code, "context": context})

    headings = [
        {"level": h["level"], "text": clean_text(h["text"])}
        for h in raw.get("section_headings", [])
        if clean_text(h.get("text", ""))
    ]

    notes = [
        {"type": n["type"], "text": clean_text(n["text"])}
        for n in raw.get("notes", [])
        if clean_text(n.get("text", ""))
    ]

    return {
        "url": raw["url"],
        "title": title,
        "section_headings": headings,
        "paragraphs": paragraphs,
        "code_blocks": code_blocks,
        "notes": notes,
    }


def clean_all(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
) -> list[str]:
    """
    Clean all raw JSON files and save cleaned versions to output_dir.

    Returns list of output file paths written.
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_path.glob("*.json"))
    logger.info(f"Cleaning {len(raw_files)} raw files from {raw_dir}")

    written = []
    skipped = 0

    for raw_file in tqdm(raw_files, desc="Cleaning"):
        with open(raw_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        cleaned = clean_page(raw)
        if cleaned is None:
            skipped += 1
            continue

        out_file = out_path / raw_file.name
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)

        written.append(str(out_file))

    logger.info(f"Cleaned {len(written)} pages, skipped {skipped} pages.")
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean raw documentation JSON files")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    paths = clean_all(args.raw_dir, args.output_dir)
    print(f"Cleaned {len(paths)} files → {args.output_dir}/")
