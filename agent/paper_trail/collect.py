"""Virtual-user collection against 小埋 for DeepSeek/AgentScope distillation."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
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
    user_llm_settings,
)

DATASET = ROOT / "dataset"
if str(DATASET) not in sys.path:
    sys.path.insert(0, str(DATASET))

import paper_trail_data.personas as personas_mod  # noqa: E402


def _reload_personas():
    """Pick up prompt/version edits without restarting a --loop process."""
    global personas_mod
    personas_mod = importlib.reload(personas_mod)
    return personas_mod


LEAK_MARKERS = (
    "我手头",
    "我是研",
    "我是博",
    "我的目标",
    "想补个",
    "按方法分类",
    "需要一份",
    "帮我整理",
)
COMPOUND_MARKERS = ("先别管", "先别贴", "先别整", "我就想知道", "其实主要", "等下你刚说")
SOFT_LEN = 85
HARD_LEN = 120
EMOJI_RE = __import__("re").compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def utterance_needs_retry(text: str) -> bool:
    if not text:
        return True
    if len(text) > SOFT_LEN:
        return True
    if any(marker in text for marker in LEAK_MARKERS):
        return True
    if text.count("，") >= 4:
        return True
    if sum(1 for marker in COMPOUND_MARKERS if marker in text) >= 2:
        return True
    if "先别" in text and "想" in text:
        return True
    if EMOJI_RE.search(text):
        return True
    if "**" in text or "```" in text:
        return True
    return False


class VirtualUser:
    """Cheap flash-side user that speaks like a sampled persona."""

    def __init__(self, config: dict, persona: dict, user_config: dict | None = None):
        # Agent `config` may point at local vLLM; the simulated user stays on flash.
        self.config = user_config or user_llm_settings()
        self.persona = persona
        self.system = personas_mod.persona_system_prompt(persona)
        self.client = openai.AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=self.config["base_url"],
            timeout=min(60, int(self.config.get("request_timeout", 180))),
            max_retries=1,
        )
        self.calls = []

    async def _generate(self, messages: list, *, max_tokens: int = 110) -> str:
        response = await self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.95,
            extra_body={"thinking": {"type": "disabled"}},
        )
        text = (response.choices[0].message.content or "").strip()
        usage = response.usage
        self.calls.append(
            {
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

    async def next_utterance(self, move: str, assistant_text: str = "") -> str:
        guide = personas_mod.move_instruction(move, self.persona)
        if assistant_text:
            # Keep only a short tail so the user does not feel forced to answer everything.
            context = "小埋上一轮回复（节选）：\n" + assistant_text[-900:]
        else:
            context = "对话刚开始，小埋还没说话。"
        messages = [
            {"role": "system", "content": self.system},
            {
                "role": "user",
                "content": (
                    f"{context}\n\n"
                    f"你想问/想推进的点（不用照念）：{guide}\n\n"
                    "把它说得像微信随口一句：短、碎、含糊；允许停顿和改口。"
                    "不要解释你在执行什么计划。"
                    "不要复述人设背景；不要一次把需求和目标讲清楚。"
                    "不要写成长复合句。"
                ),
            },
        ]
        text = await self._generate(messages, max_tokens=110)
        if utterance_needs_retry(text):
            retry_messages = messages + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "太长或太像说明书了。改写成更像真人微信的一句短话："
                        "最多两短句，别铺垫，别复述对方，别说清完整需求。"
                    ),
                },
            ]
            shorter = await self._generate(retry_messages, max_tokens=80)
            if shorter and (len(shorter) < len(text) or not utterance_needs_retry(shorter)):
                text = shorter
            if len(text) > HARD_LEN:
                text = text[:HARD_LEN].rstrip("，。；; ") + "…"
        # keep move label on the last recorded call for debugging
        if self.calls:
            self.calls[-1]["move"] = move
            if len(self.calls) >= 2 and "move" not in self.calls[-2]:
                self.calls[-2]["move"] = move
        return text

    async def close(self):
        await self.client.close()


QUOTA_MARKERS = (
    "insufficient",
    "balance",
    "quota",
    "billing",
    "payment",
    "exceeded your current quota",
    "欠费",
    "余额不足",
    "额度",
)


class QuotaExhausted(RuntimeError):
    """Raised when the provider reports insufficient balance/quota."""


def is_quota_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in QUOTA_MARKERS)


async def collect_one(
    persona: dict,
    *,
    config: dict | None = None,
    cache: ResponseCache | None = None,
    on_turn_done: Callable[[dict, int, str], None] | None = None,
    directory_root: Path | None = None,
    user_config: dict | None = None,
    extra_manifest: dict | None = None,
) -> dict:
    config = config or settings()
    cache = cache or ResponseCache(data_root() / "cache")
    turns = min(MAX_CHAIN_TURNS, int(persona.get("max_turns") or 3))
    moves = list(persona.get("jump_plan") or [])[:turns]
    while len(moves) < turns:
        moves.append("narrow")
    persona = {**persona, "jump_plan": moves, "max_turns": turns}
    session_id = uuid4().hex
    root = Path(directory_root) if directory_root is not None else constructed_root()
    directory = root / constructed_dirname(persona["id"], session_id)
    session = PaperSession(config, cache, session_id=session_id, directory=directory)
    if extra_manifest:
        session.manifest.update(extra_manifest)
        session.save_json("manifest.json", session.manifest)
    session.attach_persona(persona, moves)
    user = VirtualUser(config, persona, user_config=user_config)
    assistant_text = ""
    try:
        for index, move in enumerate(moves, start=1):
            try:
                utterance = await user.next_utterance(move, assistant_text)
                events = [event async for event in session.stream_chat(utterance)]
            except BaseException as exc:
                if is_quota_error(exc):
                    raise QuotaExhausted(str(exc)) from exc
                raise
            done = next((event for event in events if event.get("type") == "done"), None)
            if done is None:
                error = next(
                    (event for event in events if event.get("type") == "error"), None
                )
                message = (error or {}).get("message") or "采集失败"
                if is_quota_error(RuntimeError(message)):
                    raise QuotaExhausted(message)
                raise RuntimeError(message)
            assistant_text = done.get("answer") or ""
            if on_turn_done is not None:
                on_turn_done(persona, index, move)
        session.save_json("virtual_user_calls.json", user.calls)
        return {
            "session_id": session.id,
            "directory": str(session.directory),
            "storage_kind": (extra_manifest or {}).get("storage_kind") or "constructed",
            "persona_id": persona["id"],
            "prompt_version": persona.get("prompt_version")
            or personas_mod.VIRTUAL_USER_PROMPT_VERSION,
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
    personas = _reload_personas()
    config = settings()
    cache = ResponseCache(data_root() / "cache")
    version = personas.VIRTUAL_USER_PROMPT_VERSION
    sampled = personas.sample_batch(
        count, seed=seed, category=category, max_turns=max_turns
    )
    planned_turns = sum(int(p.get("max_turns") or max_turns or 4) for p in sampled)
    tqdm.write(
        f"开始采集：{len(sampled)} 条会话 × 约 {planned_turns} 轮 · "
        f"model={config['model']} · seed={seed} · concurrency={concurrency} · "
        f"prompt={version}"
    )
    started = time.monotonic()
    results = [None] * len(sampled)
    session_bar = tqdm(total=len(sampled), desc="会话", unit="条", dynamic_ncols=True)
    turn_bar = tqdm(total=planned_turns, desc="轮次", unit="轮", dynamic_ncols=True)
    progress_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)
    stop_error: BaseException | None = None

    def note_turn(persona: dict, index: int, move: str) -> None:
        turn_bar.update(1)
        turn_bar.set_postfix_str(f"{persona['id'][:24]} · {index}/{persona['max_turns']} · {move}")

    async def run_one(index: int, persona: dict) -> None:
        nonlocal stop_error
        if stop_error is not None:
            return
        async with semaphore:
            if stop_error is not None:
                return
            async with progress_lock:
                session_bar.set_postfix_str(f"运行中 {persona['id'][:32]}", refresh=False)
            try:
                result = await collect_one(
                    persona,
                    config=config,
                    cache=cache,
                    on_turn_done=note_turn,
                )
            except BaseException as exc:
                stop_error = exc
                tqdm.write(f"✗ [{index + 1}/{len(sampled)}] {persona['id']} 失败：{exc}")
                raise
            results[index] = result
            async with progress_lock:
                session_bar.update(1)
                tqdm.write(
                    f"✓ [{session_bar.n:.0f}/{len(sampled)}] {result['persona_id']} · "
                    f"{result['turns']} 轮 → {result['directory']}"
                )

    try:
        await asyncio.gather(
            *(run_one(index, persona) for index, persona in enumerate(sampled))
        )
    finally:
        session_bar.close()
        turn_bar.close()
    batch_dir = data_root() / "collections"
    batch_dir.mkdir(parents=True, exist_ok=True)
    path = batch_dir / f"batch-{seed if seed is not None else 'rand'}-{version}.json"
    payload = {
        "prompt_version": version,
        "seed": seed,
        "count": len(sampled),
        "max_turns": max_turns,
        "concurrency": concurrency,
        "model": config["model"],
        "results": results,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    try:
        if str(ROOT / "dataset") not in sys.path:
            sys.path.insert(0, str(ROOT / "dataset"))
        from paper_trail_data.quality import summarize

        report = summarize(constructed_root())
        report_path = batch_dir / "quality_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        version_stats = (report.get("by_version") or {}).get(version) or {}
        tqdm.write(
            f"质量[{version}] sessions={version_stats.get('sessions')} "
            f"open_mean={version_stats.get('open_len_mean')} "
            f"long={version_stats.get('sessions_with_long')} "
            f"compound={version_stats.get('sessions_with_compound')} "
            f"leak={version_stats.get('sessions_with_leak')}"
        )
    except Exception as exc:  # noqa: BLE001 - quality report must not fail collection
        tqdm.write(f"质量报告更新失败：{exc}")
    elapsed = time.monotonic() - started
    tqdm.write(
        f"完成：{len([r for r in results if r is not None])}/{len(sampled)} 条 · "
        f"{elapsed:.1f}s · 汇总 {path}"
    )
    if stop_error is not None:
        raise stop_error
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
    parser.add_argument(
        "--loop",
        action="store_true",
        help="循环采集：每批结束后自动开下一批，直到额度/余额不足",
    )
    args = parser.parse_args(argv)
    if args.dry_sample:
        _reload_personas()
        people = personas_mod.sample_batch(
            args.count, seed=args.seed, category=args.category, max_turns=args.max_turns
        )
        print(json.dumps(people, ensure_ascii=False, indent=2))
        return 0

    batch_index = 0
    while True:
        seed = args.seed if not args.loop else (args.seed + batch_index)
        try:
            results = asyncio.run(
                collect_batch(
                    count=args.count,
                    seed=seed,
                    category=args.category,
                    max_turns=args.max_turns,
                    concurrency=args.concurrency,
                )
            )
        except QuotaExhausted as exc:
            tqdm.write(f"额度/余额不足，停止采集：{exc}")
            return 2
        except KeyboardInterrupt:
            tqdm.write("收到中断，停止采集。")
            return 130
        except Exception as exc:
            if is_quota_error(exc):
                tqdm.write(f"额度/余额不足，停止采集：{exc}")
                return 2
            if not args.loop:
                raise
            tqdm.write(f"本批失败，稍后继续：{exc}")
            time.sleep(5)
            batch_index += 1
            continue
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        if not args.loop:
            return 0
        batch_index += 1
        tqdm.write(f"进入下一批（batch_index={batch_index}）…")


if __name__ == "__main__":
    raise SystemExit(main())
