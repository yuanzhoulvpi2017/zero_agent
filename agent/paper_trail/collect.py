"""Virtual-user collection against 小埋 for DeepSeek/AgentScope distillation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Callable
from uuid import uuid4

import openai
from dotenv import load_dotenv
from tqdm import tqdm

from paper_trail.cache import ResponseCache
from paper_trail.dialogue import MAX_CHAIN_TURNS
from paper_trail.runtime import (
    ROOT,
    PaperSession,
    constructed_dirname,
    constructed_root,
    data_root,
    settings,
)

DATASET = ROOT / "dataset"
if str(DATASET) not in sys.path:
    sys.path.insert(0, str(DATASET))

from paper_trail_data.personas import (  # noqa: E402
    move_instruction,
    persona_system_prompt,
    sample_batch,
)


class VirtualUser:
    """Cheap flash-side user that speaks like a sampled persona."""

    def __init__(self, config: dict, persona: dict):
        self.config = config
        self.persona = persona
        self.system = persona_system_prompt(persona)
        self.client = openai.AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=config["base_url"],
            timeout=min(60, int(config.get("request_timeout", 180))),
            max_retries=1,
        )
        self.calls = []

    async def next_utterance(self, move: str, assistant_text: str = "") -> str:
        guide = move_instruction(move, self.persona)
        if assistant_text:
            context = "小埋上一轮回复（节选）：\n" + assistant_text[:2000]
        else:
            context = "对话刚开始，小埋还没说话。"
        messages = [
            {"role": "system", "content": self.system},
            {
                "role": "user",
                "content": (
                    f"{context}\n\n"
                    f"你想问/想推进的点（不用照念）：{guide}\n\n"
                    "把它说得像随口聊天：短、碎、含糊；允许停顿和改口。"
                    "不要解释你在执行什么计划。"
                ),
            },
        ]
        response = await self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            max_tokens=170,
            temperature=0.95,
            extra_body={"thinking": {"type": "disabled"}},
        )
        text = (response.choices[0].message.content or "").strip()
        usage = response.usage
        self.calls.append(
            {
                "move": move,
                "request_messages": messages,
                "response_text": text,
                "usage": {
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
            }
        )
        return text

    async def close(self):
        await self.client.close()


async def collect_one(
    persona: dict,
    *,
    config: dict | None = None,
    cache: ResponseCache | None = None,
    on_turn_done: Callable[[dict, int, str], None] | None = None,
) -> dict:
    config = config or settings()
    cache = cache or ResponseCache(data_root() / "cache")
    turns = min(MAX_CHAIN_TURNS, int(persona.get("max_turns") or 3))
    moves = list(persona.get("jump_plan") or [])[:turns]
    while len(moves) < turns:
        moves.append("narrow")
    persona = {**persona, "jump_plan": moves, "max_turns": turns}
    session_id = uuid4().hex
    directory = constructed_root() / constructed_dirname(persona["id"], session_id)
    session = PaperSession(config, cache, session_id=session_id, directory=directory)
    session.attach_persona(persona, moves)
    user = VirtualUser(config, persona)
    assistant_text = ""
    try:
        for index, move in enumerate(moves, start=1):
            utterance = await user.next_utterance(move, assistant_text)
            events = [event async for event in session.stream_chat(utterance)]
            done = next((event for event in events if event.get("type") == "done"), None)
            if done is None:
                error = next(
                    (event for event in events if event.get("type") == "error"), None
                )
                raise RuntimeError((error or {}).get("message") or "采集失败")
            assistant_text = done.get("answer") or ""
            if on_turn_done is not None:
                on_turn_done(persona, index, move)
        session.save_json("virtual_user_calls.json", user.calls)
        return {
            "session_id": session.id,
            "directory": str(session.directory),
            "storage_kind": "constructed",
            "persona_id": persona["id"],
            "turns": len(moves),
            "model": config["model"],
            "dialogue_chain": str(session.directory / "dialogue_chain.json"),
        }
    finally:
        await user.close()
        await session.close()


async def collect_batch(
    count: int = 1,
    seed: int | None = 7,
    category: str | None = None,
    max_turns: int | None = 4,
    concurrency: int = 2,
) -> list:
    if max_turns is not None and not 1 <= max_turns <= MAX_CHAIN_TURNS:
        raise ValueError(f"max_turns 应为 1–{MAX_CHAIN_TURNS}")
    if concurrency < 1:
        raise ValueError("concurrency 至少为 1")
    load_dotenv(ROOT / "configs/paper_trail/.env", override=False)
    config = settings()
    cache = ResponseCache(data_root() / "cache")
    personas = sample_batch(count, seed=seed, category=category, max_turns=max_turns)
    planned_turns = sum(int(p.get("max_turns") or max_turns or 4) for p in personas)
    tqdm.write(
        f"开始采集：{len(personas)} 条会话 × 约 {planned_turns} 轮 · "
        f"model={config['model']} · seed={seed} · concurrency={concurrency}"
    )
    started = time.monotonic()
    results = [None] * len(personas)
    session_bar = tqdm(total=len(personas), desc="会话", unit="条", dynamic_ncols=True)
    turn_bar = tqdm(total=planned_turns, desc="轮次", unit="轮", dynamic_ncols=True)
    progress_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)

    def note_turn(persona: dict, index: int, move: str) -> None:
        turn_bar.update(1)
        turn_bar.set_postfix_str(f"{persona['id'][:24]} · {index}/{persona['max_turns']} · {move}")

    async def run_one(index: int, persona: dict) -> None:
        async with semaphore:
            async with progress_lock:
                session_bar.set_postfix_str(f"运行中 {persona['id'][:32]}", refresh=False)
            result = await collect_one(
                persona,
                config=config,
                cache=cache,
                on_turn_done=note_turn,
            )
            results[index] = result
            async with progress_lock:
                session_bar.update(1)
                tqdm.write(
                    f"✓ [{session_bar.n:.0f}/{len(personas)}] {result['persona_id']} · "
                    f"{result['turns']} 轮 → {result['directory']}"
                )

    try:
        await asyncio.gather(
            *(run_one(index, persona) for index, persona in enumerate(personas))
        )
    finally:
        session_bar.close()
        turn_bar.close()
    batch_dir = data_root() / "collections"
    batch_dir.mkdir(parents=True, exist_ok=True)
    path = batch_dir / f"batch-{seed if seed is not None else 'rand'}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    elapsed = time.monotonic() - started
    tqdm.write(
        f"完成：{len([r for r in results if r is not None])}/{len(personas)} 条 · "
        f"{elapsed:.1f}s · 汇总 {path}"
    )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="用虚拟人采集 PaperTrail 蒸馏对话")
    parser.add_argument("--count", type=int, default=1, help="对话条数，默认 1，省额度")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--category", default=None, help="限定角色类别 id")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=4,
        help=f"每条对话轮数上限，1–{MAX_CHAIN_TURNS}，默认 4",
    )
    parser.add_argument(
        "--dry-sample",
        action="store_true",
        help="只打印采样到的虚拟人，不调用模型",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="结束后额外打印完整 JSON 结果（默认只显示进度条与摘要）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="并行会话数，默认 2；建议 2–3，避免打满额度和限流",
    )
    args = parser.parse_args(argv)
    if args.dry_sample:
        people = sample_batch(
            args.count, seed=args.seed, category=args.category, max_turns=args.max_turns
        )
        print(json.dumps(people, ensure_ascii=False, indent=2))
        return 0
    results = asyncio.run(
        collect_batch(
            count=args.count,
            seed=args.seed,
            category=args.category,
            max_turns=args.max_turns,
            concurrency=args.concurrency,
        )
    )
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
