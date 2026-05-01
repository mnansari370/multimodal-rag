"""
Gradio demo for the Multimodal RAG Troubleshooting System.

Run with:
    python demo/app.py
    python demo/app.py --config configs/pipeline.yaml --port 7860 --share
"""

import sys
import logging
import argparse
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MultimodalRAGPipeline, PipelineConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

_pipeline: MultimodalRAGPipeline = None


# ──────────────────────────────── CSS ────────────────────────────────────────

CSS = """
/* ── Global ── */
body { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #0f2744 0%, #1a4a8a 60%, #2563b0 100%);
    border-radius: 12px;
    padding: 28px 32px 22px;
    margin-bottom: 4px;
    box-shadow: 0 4px 20px rgba(15,39,68,0.35);
}
.header-banner h1 {
    color: #ffffff;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 6px;
    letter-spacing: -0.3px;
}
.header-banner p {
    color: #b8d4f5;
    font-size: 0.97rem;
    margin: 0 0 18px;
}
.stat-pills { display: flex; gap: 10px; flex-wrap: wrap; }
.stat-pill {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 20px;
    padding: 4px 13px;
    color: #e2eeff;
    font-size: 0.8rem;
    font-weight: 500;
}

/* ── Input column ── */
.input-card {
    background: #f8fafd;
    border: 1px solid #dce8f8;
    border-radius: 10px;
    padding: 16px;
}
.how-it-works {
    background: #f0f6ff;
    border-left: 3px solid #2563b0;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #2d4a6e;
    line-height: 1.65;
}
.how-it-works b { color: #1a4a8a; }

/* ── Submit button ── */
.submit-btn {
    background: linear-gradient(90deg, #1a4a8a, #2563b0) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 11px !important;
    font-size: 1rem !important;
    letter-spacing: 0.2px !important;
    box-shadow: 0 2px 8px rgba(26,74,138,0.3) !important;
    transition: opacity 0.15s !important;
}
.submit-btn:hover { opacity: 0.88 !important; }

/* ── Answer tab ── */
.answer-box {
    background: #ffffff;
    border: 1px solid #dce8f8;
    border-radius: 10px;
    padding: 20px 22px;
    min-height: 200px;
    line-height: 1.7;
    font-size: 0.95rem;
}

/* ── Sources ── */
.source-card {
    border: 1px solid #e2eaf5;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    background: #fafcff;
}
.source-card-header { font-weight: 600; color: #1a4a8a; font-size: 0.93rem; }
.source-meta { color: #6b7f99; font-size: 0.8rem; margin: 3px 0 8px; }
.source-preview {
    background: #f4f7fb;
    border-left: 3px solid #93b9e8;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    color: #374151;
    font-family: 'Menlo', 'Consolas', monospace;
    white-space: pre-wrap;
    overflow-wrap: break-word;
}

/* ── Stats tab ── */
.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 14px;
}
.stats-card {
    background: #f8fafd;
    border: 1px solid #dce8f8;
    border-radius: 8px;
    padding: 14px 16px;
}
.stats-card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #6b7f99;
    margin-bottom: 8px;
}
.stats-card-value { font-size: 1.05rem; font-weight: 600; color: #1a4a8a; }
.stats-card-sub { font-size: 0.8rem; color: #6b7f99; margin-top: 2px; }
.stage-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #eef2f8;
    font-size: 0.87rem;
}
.stage-row:last-child { border-bottom: none; }
.stage-name { color: #374151; }
.stage-time { font-weight: 600; color: #2563b0; font-size: 0.85rem; }
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-blue { background: #dbeafe; color: #1a4a8a; }
.badge-green { background: #d1fae5; color: #065f46; }
.badge-purple { background: #ede9fe; color: #5b21b6; }
.funnel-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.88rem;
    padding: 6px 0;
}
.funnel-num { font-weight: 700; color: #1a4a8a; font-size: 1.05rem; min-width: 32px; }
.funnel-arrow { color: #93b9e8; font-size: 1.1rem; }
.funnel-label { color: #6b7f99; }
.vlp-box {
    background: #fdf4ff;
    border: 1px solid #e9d5ff;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
.vlp-title { font-weight: 600; color: #6b21a8; font-size: 0.88rem; margin-bottom: 8px; }
.vlp-field { font-size: 0.83rem; color: #374151; margin: 3px 0; }
.vlp-field span { font-weight: 500; color: #6b21a8; }

/* ── Tabs ── */
.tab-nav button { font-weight: 500 !important; }
"""

# ──────────────────────────────── Helpers ────────────────────────────────────

def _bar(value_pct: float, width: int = 120) -> str:
    filled = int(round(value_pct / 100 * width))
    return "█" * filled + "░" * (width - filled)


def _fmt_sources(chunks: list[dict]) -> str:
    if not chunks:
        return "<p style='color:#6b7f99;font-size:0.9rem;'>No sources retrieved.</p>"

    parts = []
    for i, chunk in enumerate(chunks[:5], 1):
        title = chunk.get("title", "Documentation")
        section = chunk.get("section", "")
        url = chunk.get("source_url", "")
        score = chunk.get("reranker_score", chunk.get("rrf_score", 0.0))
        preview = chunk.get("text", "")[:350]
        if len(chunk.get("text", "")) > 350:
            preview += "..."

        header_label = f"[Source {i}] {title}"
        if section and section != title:
            header_label += f" — {section}"

        score_badge = f'<span class="badge badge-blue">score {score:.3f}</span>'

        link_html = f'<a href="{url}" target="_blank" style="color:#2563b0;font-size:0.8rem;">{url}</a>' if url else ""

        parts.append(f"""
<div class="source-card">
  <div class="source-card-header">{header_label} &nbsp; {score_badge}</div>
  <div class="source-meta">{link_html}</div>
  <div class="source-preview">{preview}</div>
</div>""")

    return "\n".join(parts)


def _fmt_stats(result) -> str:
    if result is None:
        return "<p style='color:#6b7f99;'>Run a query to see pipeline stats.</p>"

    lat = result.latency
    stats = result.pruning_stats
    ref = result.reformulation
    vlp = result.vlp_output
    gen = result.generator_output

    html_parts = []

    # VLP section (only when image was provided)
    if vlp and (vlp.error_message or vlp.keywords):
        kw_str = ", ".join(vlp.keywords[:8]) if vlp.keywords else "—"
        comp_str = ", ".join(vlp.software_components) if vlp.software_components else "—"
        html_parts.append(f"""
<div class="vlp-box">
  <div class="vlp-title">Vision-Language Parser Output</div>
  <div class="vlp-field"><span>Category:</span> {vlp.visual_category or "—"}</div>
  <div class="vlp-field"><span>Error detected:</span> {vlp.error_message or "—"}</div>
  <div class="vlp-field"><span>Components:</span> {comp_str}</div>
  <div class="vlp-field"><span>Keywords:</span> {kw_str}</div>
</div>""")

    # Query reformulation
    if ref:
        strategy_label = ref.strategy if hasattr(ref, "strategy") else "adaptive"
        eff_q = ref.reformulated_query if ref.reformulated_query else "—"
        html_parts.append(f"""
<div class="stats-card" style="margin-bottom:14px;">
  <div class="stats-card-title">Query Reformulation</div>
  <div style="font-size:0.82rem;color:#6b7f99;margin-bottom:4px;">
    Strategy: <span class="badge badge-purple">{strategy_label}</span>
  </div>
  <div style="font-size:0.85rem;color:#374151;font-style:italic;line-height:1.5;">{eff_q[:220]}</div>
</div>""")

    # Retrieval funnel
    n_retrieved = len(result.retrieved_chunks)
    n_reranked = len(result.reranked_chunks)
    n_selected = len(result.selected_chunks)
    html_parts.append(f"""
<div class="stats-card" style="margin-bottom:14px;">
  <div class="stats-card-title">Retrieval Funnel</div>
  <div class="funnel-row">
    <span class="funnel-num">{n_retrieved}</span>
    <span class="funnel-label">chunks retrieved (BM25 + dense fusion)</span>
  </div>
  <div class="funnel-row">
    <span class="funnel-arrow">↓</span>
    <span class="funnel-num">{n_reranked}</span>
    <span class="funnel-label">after cross-encoder reranking</span>
  </div>
  <div class="funnel-row">
    <span class="funnel-arrow">↓</span>
    <span class="funnel-num">{n_selected}</span>
    <span class="funnel-label">sent to generator (MMR coverage pruning)</span>
  </div>
</div>""")

    # Token efficiency
    orig = stats.get("original_tokens", 0)
    pruned = stats.get("pruned_tokens", 0)
    reduction = stats.get("token_reduction_pct", 0.0)
    prompt_tok = gen.prompt_tokens if gen else 0
    completion_tok = gen.completion_tokens if gen else 0
    html_parts.append(f"""
<div class="stats-grid">
  <div class="stats-card">
    <div class="stats-card-title">Context Tokens</div>
    <div class="stats-card-value">{pruned:,}</div>
    <div class="stats-card-sub">down from {orig:,} ({reduction:.0f}% reduction)</div>
  </div>
  <div class="stats-card">
    <div class="stats-card-title">API Token Usage</div>
    <div class="stats-card-value">{prompt_tok + completion_tok:,}</div>
    <div class="stats-card-sub">{prompt_tok:,} prompt · {completion_tok:,} completion</div>
  </div>
</div>""")

    # Latency breakdown
    if lat:
        stages = [
            ("VLP (screenshot parsing)", lat.vlp_time_s),
            ("Query reformulation", lat.reformulation_time_s),
            ("Hybrid retrieval", lat.retrieval_time_s),
            ("Cross-encoder reranking", lat.reranking_time_s),
            ("Context pruning (MMR)", lat.pruning_time_s),
            ("Answer generation (Claude)", lat.generation_time_s),
        ]
        rows = "".join(
            f'<div class="stage-row"><span class="stage-name">{name}</span>'
            f'<span class="stage-time">{t:.3f}s</span></div>'
            for name, t in stages if t > 0
        )
        html_parts.append(f"""
<div class="stats-card">
  <div class="stats-card-title">Latency Breakdown</div>
  {rows}
  <div class="stage-row" style="margin-top:6px;border-top:2px solid #dce8f8;padding-top:8px;">
    <span style="font-weight:600;color:#374151;">Total</span>
    <span class="stage-time" style="font-size:0.95rem;">{lat.total_time_s:.2f}s</span>
  </div>
</div>""")

    return "\n".join(html_parts)


# ──────────────────────────────── Pipeline glue ──────────────────────────────

def load_pipeline(config_path: str = None) -> MultimodalRAGPipeline:
    cfg = PipelineConfig()
    if config_path and Path(config_path).exists():
        import yaml
        data = yaml.safe_load(open(config_path)) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    pipe = MultimodalRAGPipeline(cfg)
    pipe.load_indexes()
    return pipe


def answer_query(question: str, image, log_snippet: str):
    """Gradio handler — returns (answer_html, sources_html, stats_html)."""
    global _pipeline

    empty = ("<p style='color:#6b7f99;'>—</p>",) * 3

    if _pipeline is None:
        msg = "<p style='color:#c0392b;'>Pipeline not loaded. Restart the demo with a valid index.</p>"
        return msg, *empty[1:]

    if not question or not question.strip():
        return "<p style='color:#c0392b;'>Please enter a question.</p>", *empty[1:]

    try:
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image).convert("RGB") if image is not None else None

        result = _pipeline.run(
            question=question.strip(),
            image=pil_image,
            log_snippet=log_snippet or "",
        )
    except Exception as e:
        logger.exception("Pipeline error")
        return f"<p style='color:#c0392b;'>Error: {e}</p>", *empty[1:]

    answer_html = f'<div class="answer-box">{_md_to_html(result.answer)}</div>'
    sources_html = _fmt_sources(result.selected_chunks)
    stats_html = _fmt_stats(result)

    return answer_html, sources_html, stats_html


def _md_to_html(text: str) -> str:
    """Minimal Markdown → HTML for the answer box (bold, bullets, newlines)."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    lines = text.split("\n")
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith("- "):
            out.append(f"<li>{s[2:]}</li>")
        elif s:
            out.append(f"<p style='margin:6px 0;'>{s}</p>")
    html = "\n".join(out)
    html = re.sub(r"(<li>.*?</li>\n?)+", lambda m: f"<ul style='margin:6px 0 10px 18px;'>{m.group()}</ul>", html, flags=re.DOTALL)
    return html


# ──────────────────────────────── Interface ──────────────────────────────────

EXAMPLES = [
    ["Why is my training crashing with CUDA out of memory?", None,
     "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 10.76 GiB total capacity; 8.91 GiB already allocated)"],
    ["How do I fix NaN loss during training?", None,
     "loss went to nan after epoch 3\noptimizer: Adam lr=1e-3\nscaler: GradScaler enabled"],
    ["How do I enable mixed precision training in PyTorch?", None, ""],
    ["What causes BatchNorm to behave differently at inference?", None,
     "model.train()\nbn_layer = nn.BatchNorm2d(64)\n# outputs differ between train and eval"],
    ["My DataLoader is the bottleneck — workers seem idle. How do I fix it?", None,
     "num_workers=0\npin_memory=False\nbatch_size=128"],
    ["How do I speed up multi-GPU training with DDP?", None,
     "RuntimeError: Expected all tensors to be on the same device"],
]

BANNER_HTML = """
<div class="header-banner">
  <h1>Multimodal RAG Troubleshooter</h1>
  <p>
    Upload a screenshot, paste a log snippet, or just ask a question —
    get a grounded, cited answer from PyTorch documentation.
  </p>
  <div class="stat-pills">
    <span class="stat-pill">Hybrid BM25 + Dense Retrieval</span>
    <span class="stat-pill">Cross-Encoder Reranking</span>
    <span class="stat-pill">MMR Context Pruning</span>
    <span class="stat-pill">Claude Haiku Generation</span>
    <span class="stat-pill">MRR@10 = 0.475</span>
    <span class="stat-pill">75% Token Reduction</span>
  </div>
</div>
"""

HOW_IT_WORKS_HTML = """
<div class="how-it-works">
  <b>How this works</b><br><br>
  <b>1. Visual parsing</b> — if you upload a screenshot, a vision model extracts
  the error message, components, and keywords before retrieval.<br><br>
  <b>2. Query reformulation</b> — your question is enriched with visual cues and
  log context, so retrieval finds the right chunks even for vague queries.<br><br>
  <b>3. Hybrid retrieval</b> — BM25 (keyword) and dense (semantic) results are
  fused with Reciprocal Rank Fusion across ~10 k PyTorch documentation chunks.<br><br>
  <b>4. Cross-encoder reranking</b> — a MiniLM cross-encoder re-scores the top 50
  candidates and keeps the best 20.<br><br>
  <b>5. MMR pruning</b> — Maximum Marginal Relevance selects 5 diverse, high-relevance
  chunks, cutting context tokens by ~75%.<br><br>
  <b>6. Cited generation</b> — Claude produces a structured answer grounded in those
  5 chunks, with [Source N] citations tied to real documentation URLs.
</div>
"""


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Multimodal RAG Troubleshooter", css=CSS) as demo:

        gr.HTML(BANNER_HTML)

        with gr.Row(equal_height=False):

            # ── Left column: inputs ──────────────────────────────────────
            with gr.Column(scale=1, min_width=340):
                image_input = gr.Image(
                    label="Screenshot (optional)",
                    type="numpy",
                    height=220,
                    elem_classes=["input-card"],
                )
                question_input = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. Why does my loss go to NaN after a few steps?",
                    lines=3,
                )
                log_input = gr.Textbox(
                    label="Log / config snippet (optional)",
                    placeholder="Paste error output, stack trace, or YAML config here...",
                    lines=5,
                )
                submit_btn = gr.Button(
                    "Get Answer →",
                    variant="primary",
                    elem_classes=["submit-btn"],
                )
                gr.HTML(HOW_IT_WORKS_HTML)

            # ── Right column: outputs ────────────────────────────────────
            with gr.Column(scale=2):
                with gr.Tabs(elem_classes=["tab-nav"]):

                    with gr.TabItem("Answer"):
                        answer_output = gr.HTML(
                            value="<div class='answer-box' style='color:#6b7f99;'>Your answer will appear here.</div>"
                        )

                    with gr.TabItem("Sources"):
                        sources_output = gr.HTML(
                            value="<p style='color:#6b7f99;padding:12px;'>Source documents will appear here after you submit a query.</p>"
                        )

                    with gr.TabItem("Pipeline Stats"):
                        stats_output = gr.HTML(
                            value="<p style='color:#6b7f99;padding:12px;'>Retrieval funnel, token efficiency, and latency breakdown will appear here after you submit a query.</p>"
                        )

        submit_btn.click(
            fn=answer_query,
            inputs=[question_input, image_input, log_input],
            outputs=[answer_output, sources_output, stats_output],
        )

        gr.Examples(
            examples=EXAMPLES,
            inputs=[question_input, image_input, log_input],
            label="Example queries",
        )

    return demo


# ──────────────────────────────── Entry point ────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    logger.info("Loading pipeline...")
    _pipeline = load_pipeline(args.config)
    logger.info("Pipeline ready.")

    demo = build_interface()
    demo.launch(server_port=args.port, share=args.share)
