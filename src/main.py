"""Chrysalis Main Entry Point

Supports starting different service modes via command line.
"""

import sys
from pathlib import Path

# Ensure project root directory is in Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import click


@click.group()
def main():
    """Chrysalis - AI Personality Evolution System"""
    pass


@main.command()
def chat():
    """Start CLI interactive conversation"""
    from src.interfaces.cli import chat as cli_chat
    cli_chat.callback()


@main.command()
@click.option("--host", default="0.0.0.0", help="Server address")
@click.option("--port", default=8080, help="Server port")
def api(host: str, port: int):
    """Start REST API service"""
    from src.interfaces.api_service import run_server
    run_server(host, port)


@main.command()
@click.option("--host", default="0.0.0.0", help="Server address")
@click.option("--port", default=7860, help="Server port")
@click.option("--share", is_flag=True, help="Create public link")
def web(host: str, port: int, share: bool):
    """Start Gradio Web interface"""
    from src.interfaces.web_ui import run_web_ui
    run_web_ui(host, port, share)


@main.command()
def personality():
    """View current personality status"""
    from src.interfaces.cli import personality as cli_personality
    cli_personality.callback()


if __name__ == "__main__":
    main()
