"""
Download and clean PyTorch documentation pages.

Crawls the PyTorch docs sitemap to get all page URLs, then fetches
each page and extracts structured content: title, section headings,
paragraph text, code blocks, and warning/note boxes.
"""

import os
import json
import time
import hashlib
import logging
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# The two sitemaps we care about for PyTorch stable docs
PYTORCH_SITEMAPS = [
    "https://pytorch.org/docs/stable/sitemap.xml",
    "https://docs.pytorch.org/docs/stable/sitemap.xml",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MultimodalRAGBot/1.0)"
}


def fetch_sitemap_urls(sitemap_url: str) -> list[str]:
    """Pull all page URLs from a sitemap.xml."""
    try:
        resp = requests.get(sitemap_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Could not fetch sitemap {sitemap_url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml-xml")
    urls = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
    logger.info(f"Found {len(urls)} URLs in sitemap: {sitemap_url}")
    return urls


def extract_page_content(html: str, url: str) -> dict | None:
    """
    Parse a PyTorch documentation HTML page and extract structured content.

    Returns a dict with: url, title, section_headings, paragraphs,
    code_blocks, and notes. Returns None if the page looks empty or
    is not a real documentation page.
    """
    soup = BeautifulSoup(html, "lxml")

    # Skip pages that are just index listings
    main = soup.find("div", {"class": "body"}) or soup.find("article") or soup.find("main")
    if main is None:
        return None

    # Title
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else "Untitled"

    # Remove navigation, footer, header, and sidebar clutter
    for tag in soup.find_all(["nav", "footer", "header", "script", "style"]):
        tag.decompose()
    for tag in soup.find_all(class_=["sphinxsidebar", "related", "footer", "headerlink"]):
        tag.decompose()

    section_headings = []
    paragraphs = []
    code_blocks = []
    notes = []

    for element in main.find_all(["h1", "h2", "h3", "h4", "p", "pre", "div", "li"]):
        tag = element.name
        cls = element.get("class", [])

        if tag in ("h1", "h2", "h3", "h4"):
            text = element.get_text(strip=True)
            if text:
                section_headings.append({"level": tag, "text": text})

        elif tag == "pre":
            code = element.get_text()
            if code.strip():
                # Try to find the surrounding explanation paragraph
                prev_p = element.find_previous_sibling("p")
                context = prev_p.get_text(strip=True) if prev_p else ""
                code_blocks.append({"code": code.strip(), "context": context})

        elif tag == "p":
            text = element.get_text(strip=True)
            if text and len(text) > 20:
                paragraphs.append(text)

        elif tag == "div" and any(c in cls for c in ["note", "warning", "tip", "admonition"]):
            text = element.get_text(strip=True)
            if text:
                box_type = next((c for c in cls if c in ["note", "warning", "tip"]), "admonition")
                notes.append({"type": box_type, "text": text})

        elif tag == "li":
            text = element.get_text(strip=True)
            if text and len(text) > 15:
                paragraphs.append(text)

    if not paragraphs and not code_blocks:
        return None

    return {
        "url": url,
        "title": title,
        "section_headings": section_headings,
        "paragraphs": paragraphs,
        "code_blocks": code_blocks,
        "notes": notes,
    }


def download_pytorch_docs(
    output_dir: str = "data/raw",
    max_pages: int = None,
    delay: float = 0.3,
) -> list[str]:
    """
    Main entry point. Downloads all PyTorch documentation pages and saves
    each one as a JSON file in output_dir.

    Args:
        output_dir: Where to save the raw JSON files.
        max_pages: Cap on number of pages (None = all).
        delay: Seconds to sleep between requests to be polite.

    Returns:
        List of file paths written.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Collect all URLs from all sitemaps
    all_urls = []
    for sitemap in PYTORCH_SITEMAPS:
        all_urls.extend(fetch_sitemap_urls(sitemap))

    # Deduplicate and filter to HTML pages only
    seen = set()
    urls = []
    for url in all_urls:
        if url in seen:
            continue
        seen.add(url)
        # Skip non-HTML resources
        if any(url.endswith(ext) for ext in [".pdf", ".zip", ".png", ".jpg"]):
            continue
        urls.append(url)

    if max_pages is not None:
        urls = urls[:max_pages]

    logger.info(f"Downloading {len(urls)} pages...")

    written = []
    failed = 0

    for url in tqdm(urls, desc="Downloading docs"):
        # Use a hash of the URL as the filename so nothing collides
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        out_file = out_path / f"page_{url_hash}.json"

        if out_file.exists():
            written.append(str(out_file))
            continue

        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.debug(f"Failed to fetch {url}: {e}")
            failed += 1
            continue

        content = extract_page_content(resp.text, url)
        if content is None:
            continue

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

        written.append(str(out_file))
        time.sleep(delay)

    logger.info(f"Done. Saved {len(written)} pages, {failed} failures.")
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download PyTorch documentation")
    parser.add_argument("--output-dir", default="data/raw", help="Where to save JSON files")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit number of pages")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between requests")
    args = parser.parse_args()

    paths = download_pytorch_docs(args.output_dir, args.max_pages, args.delay)
    print(f"\nDownloaded {len(paths)} documentation pages to {args.output_dir}/")
