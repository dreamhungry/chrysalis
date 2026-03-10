"""Gradio Web聊天界面

提供可视化的对话界面，支持实时对话、反馈和人格状态展示。
"""

import sys
from pathlib import Path

# 确保项目根目录在Python路径中
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
from typing import List, Tuple

import gradio as gr

from ..bootstrap import create_agent

logger = logging.getLogger(__name__)

# 全局Agent实例
_agent = None


def get_agent():
    """获取或创建全局Agent实例"""
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent


def chat_fn(message: str, history: List[Tuple[str, str]]) -> str:
    """对话处理函数"""
    if not message.strip():
        return ""

    agent = get_agent()
    try:
        response = agent.chat(message)
        return response
    except Exception as e:
        logger.error("Chat error: %s", e)
        return f"[错误] 无法生成响应: {e}"


def feedback_fn(score: float) -> str:
    """反馈处理函数"""
    agent = get_agent()
    try:
        normalized = score / 5.0
        agent.provide_feedback(normalized)
        return f"反馈已记录: {'⭐' * int(score)} ({normalized:.2f})"
    except Exception as e:
        return f"反馈提交失败: {e}"


def personality_fn() -> str:
    """获取人格状态"""
    agent = get_agent()
    info = agent.get_personality_info()

    lines = [f"**人格描述**: {info['description']}\n"]
    lines.append(f"**更新次数**: {info['update_count']}")
    lines.append(f"**交互次数**: {info['interaction_count']}\n")
    lines.append("**维度详情**:\n")

    for name, value in info["traits"].items():
        bar_len = int(abs(value) * 10)
        if value >= 0:
            bar = "▓" * bar_len + "░" * (10 - bar_len)
        else:
            bar = "░" * (10 - bar_len) + "▓" * bar_len
        lines.append(f"- `{name:15s}` [{bar}] `{value:+.2f}`")

    return "\n".join(lines)


def create_web_ui() -> gr.Blocks:
    """创建Gradio Web界面"""
    with gr.Blocks(
        title="Chrysalis - AI人格演化系统",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# 🦋 Chrysalis\n"
            "### AI人格演化系统 - 通过交互塑造独特人格\n"
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=chat_fn,
                    title="",
                    description="与Chrysalis对话，你的反馈将影响它的人格演化。",
                    examples=[
                        "你好，介绍一下你自己",
                        "你觉得什么是创造力？",
                        "给我讲个有趣的故事",
                        "如何看待人工智能的未来？",
                    ],
                    retry_btn=None,
                    undo_btn=None,
                )

            with gr.Column(scale=1):
                gr.Markdown("### 反馈评分")
                feedback_slider = gr.Slider(
                    minimum=0,
                    maximum=5,
                    step=0.5,
                    value=3,
                    label="为最近一次回复打分 (0-5)",
                )
                feedback_btn = gr.Button("提交反馈", variant="primary")
                feedback_output = gr.Textbox(
                    label="反馈结果", interactive=False
                )

                feedback_btn.click(
                    fn=feedback_fn,
                    inputs=[feedback_slider],
                    outputs=[feedback_output],
                )

                gr.Markdown("---")
                gr.Markdown("### 人格状态")
                personality_btn = gr.Button("刷新人格状态")
                personality_output = gr.Markdown(value="点击刷新查看人格状态")

                personality_btn.click(
                    fn=personality_fn,
                    inputs=[],
                    outputs=[personality_output],
                )

    return demo


def run_web_ui(server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False):
    """启动Web界面"""
    demo = create_web_ui()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )


if __name__ == "__main__":
    run_web_ui()
