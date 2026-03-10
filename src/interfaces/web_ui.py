"""Gradio Web Chat Interface

Provides visual conversation interface with real-time chat, feedback, and personality status display.
"""

import sys
from pathlib import Path

# Ensure project root directory is in Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
from typing import List, Tuple

import gradio as gr

from ..bootstrap import create_agent

logger = logging.getLogger(__name__)

# Global Agent instance
_agent = None


def get_agent():
    """Get or create global Agent instance"""
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent


def chat_fn(message: str, history: List[Tuple[str, str]]) -> str:
    """Conversation processing function"""
    if not message.strip():
        return ""

    agent = get_agent()
    try:
        response = agent.chat(message)
        return response
    except Exception as e:
        logger.error("Chat error: %s", e)
        return f"[Error] Unable to generate response: {e}"


def feedback_fn(score: float) -> str:
    """Feedback processing function"""
    agent = get_agent()
    try:
        normalized = score / 5.0
        agent.provide_feedback(normalized)
        return f"Feedback recorded: {'*' * int(score)} ({normalized:.2f})"
    except Exception as e:
        return f"Feedback submission failed: {e}"


def personality_fn() -> str:
    """Get personality status"""
    agent = get_agent()
    info = agent.get_personality_info()

    lines = [f"**Personality Description**: {info['description']}\n"]
    lines.append(f"**Update Count**: {info['update_count']}")
    lines.append(f"**Interaction Count**: {info['interaction_count']}\n")
    lines.append("**Dimension Details**:\n")

    for name, value in info["traits"].items():
        bar_len = int(abs(value) * 10)
        if value >= 0:
            bar = "#" * bar_len + "-" * (10 - bar_len)
        else:
            bar = "-" * (10 - bar_len) + "#" * bar_len
        lines.append(f"- `{name:15s}` [{bar}] `{value:+.2f}`")

    return "\n".join(lines)


def create_web_ui() -> gr.Blocks:
    """Create Gradio Web interface"""
    with gr.Blocks(
        title="Chrysalis - AI Personality Evolution System",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Chrysalis\n"
            "### AI Personality Evolution System - Shaping Unique Personality Through Interaction\n"
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=chat_fn,
                    title="",
                    description="Chat with Chrysalis, your feedback will influence its personality evolution.",
                    examples=[
                        "Hello, introduce yourself",
                        "What do you think creativity is?",
                        "Tell me an interesting story",
                        "What's your view on the future of AI?",
                    ],
                    retry_btn=None,
                    undo_btn=None,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Feedback Rating")
                feedback_slider = gr.Slider(
                    minimum=0,
                    maximum=5,
                    step=0.5,
                    value=3,
                    label="Rate the last response (0-5)",
                )
                feedback_btn = gr.Button("Submit Feedback", variant="primary")
                feedback_output = gr.Textbox(
                    label="Feedback Result", interactive=False
                )

                feedback_btn.click(
                    fn=feedback_fn,
                    inputs=[feedback_slider],
                    outputs=[feedback_output],
                )

                gr.Markdown("---")
                gr.Markdown("### Personality Status")
                personality_btn = gr.Button("Refresh Personality Status")
                personality_output = gr.Markdown(value="Click refresh to view personality status")

                personality_btn.click(
                    fn=personality_fn,
                    inputs=[],
                    outputs=[personality_output],
                )

    return demo


def run_web_ui(server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False):
    """Start Web interface"""
    demo = create_web_ui()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )


if __name__ == "__main__":
    run_web_ui()
