"""
Gradio demo interface for the Multimodal RAG Troubleshooting System.

Provides an interactive UI where users can:
  - Upload a screenshot (optional)
  - Type a technical question
  - Paste a log or config snippet (optional)
  - Get a cited answer with the top 3 retrieved source chunks shown

Run with:
  python demo/app.py --config configs/pipeline.yaml
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path

import gradio as gr

# Make sure src is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import MultimodalRAGPipeline, PipelineConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance — loaded once at startup
_pipeline: MultimodalRAGPipeline = None


def load_pipeline(config_path: str = None) -> MultimodalRAGPipeline:
    """Load the pipeline with optional YAML config."""
    cfg = PipelineConfig()

    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    pipeline = MultimodalRAGPipeline(cfg)
    pipeline.load_indexes()
    return pipeline


def answer_query(
    question: str,
    image,
    log_snippet: str,
) -> tuple[str, str, str]:
    """
    Gradio handler — called when user clicks Submit.

    Returns:
        (answer_text, sources_markdown, debug_info)
    """
    global _pipeline

    if _pipeline is None:
        return (
            "Pipeline not loaded. Please restart the demo with a valid index.",
            "",
            "",
        )

    if not question or not question.strip():
        return "Please enter a question.", "", ""

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
        return f"An error occurred: {str(e)}", "", ""

    # Format the answer
    answer_text = result.answer

    # Format source citations — show top 3 chunks
    top_chunks = result.selected_chunks[:3]
    source_lines = []
    for i, chunk in enumerate(top_chunks, 1):
        title = chunk.get("title", "Unknown")
        section = chunk.get("section", title)
        url = chunk.get("source_url", "")
        score = chunk.get("reranker_score", chunk.get("rrf_score", 0.0))
        preview = chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else "")
        source_lines.append(
            f"**[Source {i}] {title} — {section}**  \n"
            f"URL: {url}  \n"
            f"Score: {score:.3f}  \n"
            f"```\n{preview}\n```\n"
        )

    sources_markdown = "\n---\n".join(source_lines) if source_lines else "No sources retrieved."

    # Debug info
    stats = result.pruning_stats
    latency = result.latency
    reformulated = result.reformulation.reformulated_query if result.reformulation else question

    debug_parts = [
        f"**Reformulated query:** {reformulated[:200]}",
        f"**Retrieved:** {len(result.retrieved_chunks)} → Reranked: {len(result.reranked_chunks)} → Selected: {len(result.selected_chunks)}",
        f"**Token reduction:** {stats.get('original_tokens', '?')} → {stats.get('pruned_tokens', '?')} tokens ({stats.get('token_reduction_pct', '?')}%)",
    ]
    if latency:
        debug_parts.append(f"**Total latency:** {latency.total_time_s:.2f}s (retrieval: {latency.retrieval_time_s:.2f}s, generation: {latency.generation_time_s:.2f}s)")

    debug_info = "\n\n".join(debug_parts)

    return answer_text, sources_markdown, debug_info


def build_interface() -> gr.Blocks:
    """Build and return the Gradio interface."""
    with gr.Blocks(
        title="Multimodal RAG Troubleshooter",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
# Multimodal RAG Troubleshooter
**AI-powered technical troubleshooting from screenshots, logs, and questions.**

Upload a screenshot of an error, paste a log snippet, ask your question — and get a cited answer from the PyTorch documentation.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Screenshot (optional)",
                    type="numpy",
                    height=300,
                )
                question_input = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. Why is my training crashing?",
                    lines=3,
                )
                log_input = gr.Textbox(
                    label="Log / config snippet (optional)",
                    placeholder="Paste error output, stack trace, or YAML config here...",
                    lines=6,
                )
                submit_btn = gr.Button("Get Answer", variant="primary")

            with gr.Column(scale=2):
                answer_output = gr.Markdown(label="Answer")
                with gr.Accordion("Source Documents (top 3)", open=True):
                    sources_output = gr.Markdown()
                with gr.Accordion("Debug Info", open=False):
                    debug_output = gr.Markdown()

        submit_btn.click(
            fn=answer_query,
            inputs=[question_input, image_input, log_input],
            outputs=[answer_output, sources_output, debug_output],
        )

        gr.Examples(
            examples=[
                ["Why is my training crashing with CUDA out of memory?", None, "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"],
                ["What is wrong with my batch size setting?", None, "batch_size: 256\nmodel: resnet50\nprecision: float32"],
                ["How do I enable mixed precision training in PyTorch?", None, ""],
            ],
            inputs=[question_input, image_input, log_input],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Path to pipeline config")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    logger.info("Loading pipeline...")
    _pipeline = load_pipeline(args.config)
    logger.info("Pipeline ready.")

    demo = build_interface()
    demo.launch(server_port=args.port, share=args.share)
