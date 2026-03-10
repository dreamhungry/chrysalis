"""CLI命令行接口

提供交互式对话、用户反馈输入、人格状态查询等功能。
"""

import sys
from pathlib import Path

# 确保项目根目录在Python路径中
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import click

from ..bootstrap import create_agent


@click.group()
def cli():
    """Chrysalis - AI人格演化系统"""
    pass


@cli.command()
def chat():
    """启动交互式对话"""
    agent = create_agent()

    click.echo("=" * 60)
    click.echo("  Chrysalis - AI人格演化系统")
    click.echo("  输入消息开始对话，输入以下命令执行特殊操作：")
    click.echo("    /feedback <分数>  - 对上一条回复打分 (0-5)")
    click.echo("    /personality      - 查看当前人格状态")
    click.echo("    /history          - 查看最近对话")
    click.echo("    /quit             - 退出")
    click.echo("=" * 60)
    click.echo()

    personality_info = agent.get_personality_info()
    click.echo(f"当前人格: {personality_info['description']}")
    click.echo()

    while True:
        try:
            user_input = click.prompt("你", prompt_suffix="> ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("\n再见！")
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.startswith("/"):
            _handle_command(agent, user_input)
            continue

        # 正常对话
        try:
            response = agent.chat(user_input)
            click.echo(f"\nChrysalis> {response}\n")
        except Exception as e:
            click.echo(f"\n[错误] {e}\n")


@cli.command()
def personality():
    """查看当前人格状态"""
    agent = create_agent()
    info = agent.get_personality_info()
    _display_personality(info)


@cli.command()
@click.argument("n", default=10)
def history(n: int):
    """查看最近N条交互记录"""
    agent = create_agent()
    records = agent.memory.get_recent(n)

    if not records:
        click.echo("暂无交互记录。")
        return

    for r in records:
        click.echo(f"--- [{r.get('timestamp', '?')}] ---")
        click.echo(f"  用户: {r.get('user_input', '')}")
        click.echo(f"  助手: {r.get('agent_response', '')}")
        if r.get("feedback") is not None:
            click.echo(f"  反馈: {'⭐' * int(r['feedback'] * 5)} ({r['feedback']:.2f})")
        click.echo()


def _handle_command(agent, command: str) -> None:
    """处理CLI命令"""
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()

    if cmd == "/quit" or cmd == "/exit":
        click.echo("再见！")
        raise SystemExit(0)

    elif cmd == "/feedback":
        if len(parts) < 2:
            click.echo("用法: /feedback <0-5>")
            return
        try:
            score = float(parts[1])
            if score < 0 or score > 5:
                click.echo("分数范围: 0-5")
                return
            feedback = score / 5.0  # 归一化到 [0, 1]
            agent.provide_feedback(feedback)
            click.echo(f"反馈已记录: {'⭐' * int(score)} ({feedback:.2f})")
        except ValueError:
            click.echo("请输入有效数字 (0-5)")

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
            click.echo("暂无交互记录。")
        else:
            for r in records:
                click.echo(f"  [{r.get('timestamp', '?')}]")
                click.echo(f"    用户: {r.get('user_input', '')[:60]}...")
                click.echo(f"    助手: {r.get('agent_response', '')[:60]}...")
                click.echo()

    else:
        click.echo(f"未知命令: {cmd}")


def _display_personality(info: dict) -> None:
    """展示人格信息"""
    click.echo("\n=== 人格状态 ===")
    click.echo(f"描述: {info['description']}")
    click.echo(f"更新次数: {info['update_count']}")
    click.echo(f"交互次数: {info['interaction_count']}")
    click.echo("\n维度详情:")

    traits = info.get("traits", {})
    for name, value in traits.items():
        bar_len = int(abs(value) * 20)
        if value >= 0:
            bar = " " * 20 + "│" + "█" * bar_len + " " * (20 - bar_len)
        else:
            bar = " " * (20 - bar_len) + "█" * bar_len + "│" + " " * 20
        click.echo(f"  {name:15s} [{bar}] {value:+.2f}")
    click.echo()


def main():
    cli()


if __name__ == "__main__":
    main()
