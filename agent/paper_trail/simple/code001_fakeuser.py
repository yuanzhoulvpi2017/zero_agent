"""虚拟人与 code001 的小埋自动聊天；在终端直接观看人设和对话。

uv run --project agent agent/paper_trail/simple/code001_fakeuser.py
uv run --project agent agent/paper_trail/simple/code001_fakeuser.py --category industry_engineer --turns 5 --seed 42
"""

import argparse
import asyncio
import random

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from paper_trail.cache import ResponseCache
# collect 已复用 dataset/ 中的人设和提示词，不在本示例复制角色逻辑。
from paper_trail.collect import VirtualUser, personas_mod
from paper_trail.runtime import ROOT, load_settings
from paper_trail.simple.code001 import PaperSession

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=personas_mod.CATEGORIES, help="指定角色类别")
    parser.add_argument("--turns", type=int, default=3, help="对话轮数，默认 3，最多 20")
    parser.add_argument("--seed", type=int, help="固定人设采样种子（模型发言仍可能变化）")
    parser.add_argument("--list", action="store_true", help="列出角色模板后退出，不调用模型")
    parser.add_argument("--dry-run", action="store_true", help="只展示采样人设，不调用模型")
    args = parser.parse_args()
    if not 1 <= args.turns <= personas_mod.MAX_CHAIN_TURNS:
        parser.error(f"--turns 必须为 1–{personas_mod.MAX_CHAIN_TURNS}")
    return args


def show_persona(persona):
    table = Table(show_header=False, expand=True)
    for label, key in (
        ("角色", "category_label"), ("背景", "background"),
        ("心里目标", "goal"), ("说话风格", "voice"), ("话题", "topic"),
        ("可能岔开的话题", "side_topic"), ("逐轮动作", "jump_plan"),
    ):
        value = persona[key]
        table.add_row(label, " → ".join(value) if isinstance(value, list) else value)
    console.print(Panel(table, title=f"虚拟人：{persona['name']} · {persona['id']}"))


async def main():
    args = parse_args()
    if args.list:
        table = Table("类别", "模板 ID", "名字", "背景")
        for role in personas_mod.catalog():
            table.add_row(role["category"], role["template_id"], role["name"], role["background"])
        console.print(table)
        return

    persona = personas_mod.sample_persona(
        random.Random(args.seed), category=args.category, max_turns=args.turns,
    )
    show_persona(persona)
    if args.dry_run:
        return

    config = load_settings(ROOT / "configs/paper_trail/agent.toml")
    user = VirtualUser(config, persona)
    session = None
    dialogue = {"persona": persona, "status": "running", "turns": []}
    try:
        session = PaperSession(
            config, cache=ResponseCache(ROOT / "data/paper_trail/cache"),
        )
        model = session.agent.model
        console.print(f"记录目录：{model.directory}", markup=False)
        model.save_json(model.directory / "persona.json", persona)
        model.save_json(model.directory / "virtual_user_config.json", user.config)
        assistant_text = ""
        for index, move in enumerate(persona["jump_plan"], start=1):
            console.rule(f"第 {index}/{args.turns} 轮 · {move}")
            text = await user.next_utterance(move, assistant_text)
            if not text.strip():
                raise RuntimeError("虚拟人返回空发言，已停止演示。")
            console.print(Panel(text, title=f"虚拟人 · {persona['name']}", border_style="cyan"))
            turn = {"index": index, "move": move, "user": text, "assistant": None}
            dialogue["turns"].append(turn)
            # code001 的 Agent 自带流式输出，包含工具调用；避免重复打印回复。
            console.print("[bold green]小埋[/bold green]")
            assistant_text = await session.chat(text) or ""
            turn["assistant"] = assistant_text
            model.save_json(model.directory / "conversation.json", dialogue)
        dialogue["status"] = "completed"
    except BaseException as exc:
        dialogue["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed"
        dialogue["error_type"] = type(exc).__name__
        raise
    finally:
        try:
            if session is not None:
                model = session.agent.model
                model.save_json(model.directory / "conversation.json", dialogue)
                model.save_json(model.directory / "virtual_user_calls.json", {"calls": user.calls})
        finally:
            try:
                await user.close()
            finally:
                if session is not None:
                    await session.client.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[dim]演示已中断，已有对话记录已保留。[/dim]")
