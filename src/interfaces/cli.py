"""CLI Command Line Interface

Provides interactive conversation, user feedback input, and personality status query functions.
"""

import sys
from pathlib import Path

# Ensure project root directory is in Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import click

from ..bootstrap import create_agent


@click.group()
def cli():
    """Chrysalis - AI Personality Evolution System"""
    pass


@cli.command()
def chat():
    """Start interactive conversation"""
    agent = create_agent()

    click.echo("=" * 60)
    click.echo("  Chrysalis - AI Personality Evolution System")
    click.echo("  Enter message to start conversation, use commands below:")
    click.echo("    /feedback <score>  - Rate last response (0-5)")
    click.echo("    /personality       - View current personality status")
    click.echo("    /history           - View recent conversations")
    click.echo("    /quit              - Exit")
    click.echo("=" * 60)
    click.echo()

    personality_info = agent.get_personality_info()
    click.echo(f"Current personality: {personality_info['description']}")
    click.echo()

    while True:
        try:
            user_input = click.prompt("You", prompt_suffix="> ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            _handle_command(agent, user_input)
            continue

        # Normal conversation
        try:
            response = agent.chat(user_input)
            click.echo(f"\nChrysalis> {response}\n")
        except Exception as e:
            click.echo(f"\n[Error] {e}\n")


@cli.command()
def personality():
    """View current personality status"""
    agent = create_agent()
    info = agent.get_personality_info()
    _display_personality(info)


@cli.command()
@click.argument("n", default=10)
def history(n: int):
    """View recent N interaction records"""
    agent = create_agent()
    records = agent.memory.get_recent(n)

    if not records:
        click.echo("No interaction records available.")
        return

    for r in records:
        click.echo(f"--- [{r.get('timestamp', '?')}] ---")
        click.echo(f"  User: {r.get('user_input', '')}")
        click.echo(f"  Assistant: {r.get('agent_response', '')}")
        if r.get("feedback") is not None:
            click.echo(f"  Feedback: {'*' * int(r['feedback'] * 5)} ({r['feedback']:.2f})")
        click.echo()


def _handle_command(agent, command: str) -> None:
    """Handle CLI commands"""
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()

    if cmd == "/quit" or cmd == "/exit":
        click.echo("Goodbye!")
        raise SystemExit(0)

    elif cmd == "/feedback":
        if len(parts) < 2:
            click.echo("Usage: /feedback <0-5>")
            return
        try:
            score = float(parts[1])
            if score < 0 or score > 5:
                click.echo("Score range: 0-5")
                return
            feedback = score / 5.0  # Normalize to [0, 1]
            agent.provide_feedback(feedback)
            click.echo(f"Feedback recorded: {'*' * int(score)} ({feedback:.2f})")
        except ValueError:
            click.echo("Please enter a valid number (0-5)")

    elif cmd == "/personality":
        info = agent.get_personality_info()
        _display_personality(info)

    elif cmd == "/history":
        n = 5
        if len(parts) > 1:
            try:
                n = int(parts[1])
            except ValueError:
                pass
        records = agent.memory.get_recent(n)
        if not records:
            click.echo("No interaction records available.")
        else:
            for r in records:
                click.echo(f"  [{r.get('timestamp', '?')}]")
                click.echo(f"    User: {r.get('user_input', '')[:60]}...")
                click.echo(f"    Assistant: {r.get('agent_response', '')[:60]}...")
                click.echo()

    else:
        click.echo(f"Unknown command: {cmd}")


def _display_personality(info: dict) -> None:
    """Display personality information"""
    click.echo("\n=== Personality Status ===")
    click.echo(f"Description: {info['description']}")
    click.echo(f"Update count: {info['update_count']}")
    click.echo(f"Interaction count: {info['interaction_count']}")
    click.echo("\nDimension details:")

    traits = info.get("traits", {})
    for name, value in traits.items():
        bar_len = int(abs(value) * 20)
        if value >= 0:
            bar = " " * 20 + "|" + "=" * bar_len + " " * (20 - bar_len)
        else:
            bar = " " * (20 - bar_len) + "=" * bar_len + "|" + " " * 20
        click.echo(f"  {name:15s} [{bar}] {value:+.2f}")
    click.echo()


def main():
    cli()


if __name__ == "__main__":
    main()
