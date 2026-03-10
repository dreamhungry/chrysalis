"""Chrysalis 主入口

支持通过命令行启动不同的服务模式。
"""

import sys
from pathlib import Path

# 确保项目根目录在Python路径中
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import click


@click.group()
def main():
    """Chrysalis - AI人格演化系统"""
    pass


@main.command()
def chat():
    """启动CLI交互式对话"""
    from src.interfaces.cli import chat as cli_chat
    cli_chat.callback()


@main.command()
@click.option("--host", default="0.0.0.0", help="服务器地址")
@click.option("--port", default=8080, help="服务器端口")
def api(host: str, port: int):
    """启动REST API服务"""
    from src.interfaces.api_service import run_server
    run_server(host, port)


@main.command()
@click.option("--host", default="0.0.0.0", help="服务器地址")
@click.option("--port", default=7860, help="服务器端口")
@click.option("--share", is_flag=True, help="创建公开链接")
def web(host: str, port: int, share: bool):
    """启动Gradio Web界面"""
    from src.interfaces.web_ui import run_web_ui
    run_web_ui(host, port, share)


@main.command()
def personality():
    """查看当前人格状态"""
    from src.interfaces.cli import personality as cli_personality
    cli_personality.callback()


if __name__ == "__main__":
    main()
