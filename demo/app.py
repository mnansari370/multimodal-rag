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


# ── Theme ──────────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
)


# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
/* ── Container ── */
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #0f2744 0%, #1e40af 60%, #2563eb 100%);
    border-radius: 14px;
    padding: 26px 30px 22px;
    margin-bottom: 18px;
    box-shadow: 0 4px 20px rgba(15,39,68,0.3);
}
.app-header h1 {
    color: #fff;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 7px;
    letter-spacing: -0.3px;
}
.app-header p {
    color: #bfdbfe;
    font-size: 0.93rem;
    margin: 0 0 16px;
}
.pills { display: flex; gap: 8px; flex-wrap: wrap; }
.pill {
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 20px;
    padding: 3px 11px;
    color: #e0f2fe;
    font-size: 0.75rem;
    font-weight: 500;
}

/* ── Submit button ── */
#submit-btn, #submit-btn button {
    width: 100% !important;
    padding: 13px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    border-radius: 8px !important;
    background: linear-gradient(90deg, #1e40af, #2563eb) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.35) !important;
    margin-top: 4px !important;
    cursor: pointer !important;
}

/* ── Output containers ── */
.answer-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px 24px;
    min-height: 200px;
    line-height: 1.75;
    font-size: 0.94rem;
    color: #1e293b;
}
.answer-box p  { margin: 5px 0; }
.answer-box ul { margin: 5px 0 10px 20px; padding: 0; }
.answer-box li { margin: 4px 0; }
.answer-box strong { color: #1e3a5f; }

/* ── Source cards ── */
.src-card {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    background: #f8fafc;
}
.src-title { font-weight: 700; color: #1e3a5f; font-size: 0.9rem; margin-bottom: 3px; }
.src-meta  { font-size: 0.77rem; color: #64748b; margin-bottom: 8px; }
.src-badge {
    display: inline-block;
    background: #dbeafe;
    color: #1e40af;
    border-radius: 10px;
    padding: 1px 8px;
    font-size: 0.71rem;
    font-weight: 600;
    margin-left: 5px;
}
.src-preview {
    background: #f1f5f9;
    border-left: 3px solid #60a5fa;
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    font-size: 0.79rem;
    font-family: 'Menlo', 'Consolas', monospace;
    white-space: pre-wrap;
    color: #334155;
    overflow-wrap: break-word;
}

/* ── Stats ── */
.stat-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.stat-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #64748b;
    margin-bottom: 10px;
}
.funnel-step { display: flex; align-items: center; gap: 10px; padding: 4px 0; font-size: 0.87rem; }
.funnel-num   { font-weight: 700; color: #1d4ed8; font-size: 1.05rem; min-width: 38px; }
.funnel-arrow { color: #93c5fd; font-size: 1rem; }
.stage-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.86rem;
    color: #334155;
}
.stage-row:last-child { border-bottom: none; }
.stage-time { font-weight: 600; color: #1d4ed8; }
.mini-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }
.mini-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 12px; text-align: center; }
.mini-val   { font-size: 1.5rem; font-weight: 700; color: #1d4ed8; line-height: 1; }
.mini-label { font-size: 0.76rem; color: #64748b; margin-top: 4px; }
.vlp-box {
    background: #fdf4ff;
    border: 1px solid #e9d5ff;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.vlp-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #7e22ce; margin-bottom: 8px; }
.vlp-row   { font-size: 0.84rem; color: #374151; margin: 3px 0; }
.vlp-row strong { color: #6b21a8; }

/* ── How it works ── */
.hiw-box {
    background: #f0f9ff;
    border-left: 4px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 14px 16px;
    font-size: 0.83rem;
    line-height: 1.7;
    color: #0c4a6e;
    margin-top: 12px;
}
.hiw-box strong { color: #0369a1; }

/* ── Placeholder ── */
.placeholder {
    color: #94a3b8;
    font-size: 0.88rem;
    padding: 24px 16px;
    text-align: center;
}

/* ── Tabs ── */
.tab-wrap button { font-weight: 500 !important; }

/* ── Image upload — make it obviously clickable ── */
.image-upload-area .wrap { min-height: 160px !important; cursor: pointer !important; }
"""


# ── Static HTML blocks ─────────────────────────────────────────────────────────

HEADER_HTML = """
<div class="app-header">
  <h1>Multimodal RAG Troubleshooter</h1>
  <p>Upload a screenshot, paste a log snippet, or just ask a question — get a grounded, cited answer from PyTorch docs.</p>
  <div class="pills">
    <span class="pill">Hybrid BM25 + Dense Retrieval</span>
    <span class="pill">Cross-Encoder Reranking</span>
    <span class="pill">MMR Context Pruning</span>
    <span class="pill">Claude Haiku Generation</span>
    <span class="pill">MRR@10 = 0.475</span>
    <span class="pill">75% Token Reduction</span>
  </div>
</div>
"""

HOW_IT_WORKS_HTML = """
<div class="hiw-box">
  <strong>How it works</strong><br><br>
  <strong>1. Visual parsing</strong> — screenshot is parsed by a VLM to extract error text, components, and keywords.<br>
  <strong>2. Query reformulation</strong> — question is enriched with visual cues and log context.<br>
  <strong>3. Hybrid retrieval</strong> — BM25 + dense FAISS results fused with Reciprocal Rank Fusion over ~10 k PyTorch doc chunks.<br>
  <strong>4. Cross-encoder reranking</strong> — MiniLM re-scores top 50, keeps best 20.<br>
  <strong>5. MMR pruning</strong> — 5 diverse, high-relevance chunks selected (~75% token cut).<br>
  <strong>6. Cited generation</strong> — Claude answers with [Source N] citations from real doc URLs.
</div>
"""

PLACEHOLDER_ANSWER = "<div class='placeholder'>Your answer will appear here after you submit a query.</div>"
PLACEHOLDER_SOURCES = "<div class='placeholder'>Source documents will appear here.</div>"
PLACEHOLDER_STATS = "<div class='placeholder'>Pipeline stats — retrieval funnel, token efficiency, and latency — will appear here.</div>"

EXAMPLES = [
    ["Why is my training crashing with CUDA out of memory?", None,
     "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 10.76 GiB total; 8.91 GiB already allocated)"],
    ["How do I fix NaN loss during training?", None,
     "loss went to nan after epoch 3\noptimizer: Adam lr=1e-3\nscaler: GradScaler enabled"],
    ["How do I enable mixed precision training in PyTorch?", None, ""],
    ["What causes BatchNorm to behave differently during inference?", None, ""],
    ["My DataLoader is the bottleneck — how do I speed it up?", None,
     "num_workers=0\npin_memory=False\nbatch_size=128"],
    ["How do I train across multiple GPUs with DDP?", None,
     "RuntimeError: Expected all tensors to be on the same device"],
]


# ── Output formatters ──────────────────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    out, in_ul = [], False
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("- "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{s[2:]}</li>")
        else:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if s:
                out.append(f"<p>{s}</p>")
    if in_ul:
        out.append("</ul>")
    return "\n".join(out)


def _fmt_sources(chunks: list) -> str:
    if not chunks:
        return "<div class='placeholder'>No sources retrieved.</div>"
    parts = []
    for i, c in enumerate(chunks[:5], 1):
        title   = c.get("title", "Documentation")
        section = c.get("section", "")
        url     = c.get("source_url", "")
        score   = c.get("reranker_score", c.get("rrf_score", 0.0))
        text    = c.get("text", "")
        preview = text[:350] + ("…" if len(text) > 350 else "")

        heading = f"[{i}] {title}"
        if section and section != title:
            heading += f" — {section}"
        link_html = (f'<a href="{url}" target="_blank" style="color:#2563eb;">{url}</a>'
                     if url else "")
        parts.append(f"""
<div class="src-card">
  <div class="src-title">{heading}<span class="src-badge">score {score:.3f}</span></div>
  <div class="src-meta">{link_html}</div>
  <div class="src-preview">{preview}</div>
</div>""")
    return "\n".join(parts)


def _fmt_stats(result) -> str:
    if result is None:
        return PLACEHOLDER_STATS

    lat   = result.latency
    stats = result.pruning_stats
    ref   = result.reformulation
    vlp   = result.vlp_output
    gen   = result.generator_output
    parts = []

    # VLP
    if vlp and (vlp.error_message or vlp.keywords):
        kw   = ", ".join(vlp.keywords[:8]) or "—"
        comp = ", ".join(vlp.software_components) or "—"
        parts.append(f"""
<div class="vlp-box">
  <div class="vlp-label">Vision-Language Parser output</div>
  <div class="vlp-row"><strong>Category:</strong> {vlp.visual_category or "—"}</div>
  <div class="vlp-row"><strong>Error detected:</strong> {vlp.error_message or "—"}</div>
  <div class="vlp-row"><strong>Components:</strong> {comp}</div>
  <div class="vlp-row"><strong>Keywords:</strong> {kw}</div>
</div>""")

    # Reformulation
    if ref:
        strat = getattr(ref, "strategy", "adaptive")
        q     = (ref.reformulated_query or "—")[:240]
        parts.append(f"""
<div class="stat-card">
  <div class="stat-title">Query reformulation</div>
  <div style="font-size:0.8rem;color:#64748b;margin-bottom:6px;">
    Strategy: <strong style="color:#1d4ed8;">{strat}</strong>
  </div>
  <div style="font-size:0.85rem;color:#334155;font-style:italic;line-height:1.55;">{q}</div>
</div>""")

    # Retrieval funnel
    nr  = len(result.retrieved_chunks)
    nrk = len(result.reranked_chunks)
    ns  = len(result.selected_chunks)
    parts.append(f"""
<div class="stat-card">
  <div class="stat-title">Retrieval funnel</div>
  <div class="funnel-step"><span class="funnel-num">{nr}</span><span>candidates — hybrid BM25 + dense fusion</span></div>
  <div class="funnel-step"><span class="funnel-arrow">↓</span></div>
  <div class="funnel-step"><span class="funnel-num">{nrk}</span><span>after cross-encoder reranking</span></div>
  <div class="funnel-step"><span class="funnel-arrow">↓</span></div>
  <div class="funnel-step"><span class="funnel-num">{ns}</span><span>sent to generator (MMR pruning)</span></div>
</div>""")

    # Token + latency summary cards
    orig    = stats.get("original_tokens", 0)
    pruned  = stats.get("pruned_tokens", 0)
    red_pct = stats.get("token_reduction_pct", 0.0)
    ptok    = gen.prompt_tokens if gen else 0
    ctok    = gen.completion_tokens if gen else 0
    total_s = lat.total_time_s if lat else 0.0

    parts.append(f"""
<div class="mini-grid">
  <div class="mini-card">
    <div class="mini-val">{red_pct:.0f}%</div>
    <div class="mini-label">token reduction<br><span style="font-size:0.71rem;">{orig:,} → {pruned:,}</span></div>
  </div>
  <div class="mini-card">
    <div class="mini-val">{total_s:.1f}s</div>
    <div class="mini-label">total latency<br><span style="font-size:0.71rem;">{ptok+ctok:,} API tokens used</span></div>
  </div>
</div>""")

    # Stage latency
    if lat:
        stages = [
            ("VLP (screenshot parsing)",    lat.vlp_time_s),
            ("Query reformulation",          lat.reformulation_time_s),
            ("Hybrid retrieval",             lat.retrieval_time_s),
            ("Cross-encoder reranking",      lat.reranking_time_s),
            ("MMR context pruning",          lat.pruning_time_s),
            ("Answer generation (Claude)",   lat.generation_time_s),
        ]
        rows = "".join(
            f'<div class="stage-row"><span>{name}</span><span class="stage-time">{t:.3f}s</span></div>'
            for name, t in stages if t > 0
        )
        parts.append(f"""
<div class="stat-card">
  <div class="stat-title">Latency by stage</div>
  {rows}
  <div class="stage-row" style="margin-top:6px;border-top:2px solid #e2e8f0;padding-top:8px;">
    <strong>Total</strong>
    <span class="stage-time" style="font-size:0.95rem;">{total_s:.2f}s</span>
  </div>
</div>""")

    return "\n".join(parts)


# ── Pipeline glue ──────────────────────────────────────────────────────────────

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
    """Gradio handler — image arrives as a PIL Image (type='pil')."""
    global _pipeline

    if _pipeline is None:
        err = "<div class='answer-box' style='color:#dc2626;'>Pipeline not loaded — restart the demo.</div>"
        return err, PLACEHOLDER_SOURCES, PLACEHOLDER_STATS

    if not question or not question.strip():
        err = "<div class='answer-box' style='color:#dc2626;'>Please enter a question.</div>"
        return err, PLACEHOLDER_SOURCES, PLACEHOLDER_STATS

    try:
        result = _pipeline.run(
            question=question.strip(),
            image=image,          # PIL Image or None — pipeline.run() accepts both
            log_snippet=log_snippet or "",
        )
    except Exception as e:
        logger.exception("Pipeline error")
        err = f"<div class='answer-box' style='color:#dc2626;'>Error: {e}</div>"
        return err, PLACEHOLDER_SOURCES, PLACEHOLDER_STATS

    answer_html  = f'<div class="answer-box">{_md_to_html(result.answer)}</div>'
    sources_html = _fmt_sources(result.selected_chunks)
    stats_html   = _fmt_stats(result)
    return answer_html, sources_html, stats_html


# ── Interface ──────────────────────────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    with gr.Blocks(css=CSS, theme=THEME, title="Multimodal RAG Troubleshooter") as demo:

        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):

            # ── Left: inputs ───────────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):
                question_input = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. Why does my training loss go to NaN?",
                    lines=3,
                )
                image_input = gr.Image(
                    label="Screenshot (optional) — click here or drag & drop an image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    elem_classes=["image-upload-area"],
                )
                log_input = gr.Textbox(
                    label="Log / config snippet (optional)",
                    placeholder="Paste error output, stack trace, or YAML config here...",
                    lines=4,
                )
                submit_btn = gr.Button(
                    "Get Answer →",
                    variant="primary",
                    elem_id="submit-btn",
                )
                gr.HTML(HOW_IT_WORKS_HTML)

            # ── Right: outputs ─────────────────────────────────────────────
            with gr.Column(scale=2):
                with gr.Tabs(elem_classes=["tab-wrap"]):
                    with gr.TabItem("Answer"):
                        answer_output = gr.HTML(value=PLACEHOLDER_ANSWER)

                    with gr.TabItem("Sources"):
                        sources_output = gr.HTML(value=PLACEHOLDER_SOURCES)

                    with gr.TabItem("Pipeline Stats"):
                        stats_output = gr.HTML(value=PLACEHOLDER_STATS)

        submit_btn.click(
            fn=answer_query,
            inputs=[question_input, image_input, log_input],
            outputs=[answer_output, sources_output, stats_output],
        )

        gr.Examples(
            examples=EXAMPLES,
            inputs=[question_input, image_input, log_input],
            label="Example queries — click any row to load it",
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

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
