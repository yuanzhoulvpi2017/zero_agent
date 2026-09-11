"""Score constructed virtual-user dialogues for human-likeness heuristics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

LEAK_MARKERS = (
    "我手头",
    "我是研",
    "我是博",
    "我的目标",
    "想补个",
    "按方法分类整理",
    "需要一份",
    "帮我整理",
)
COMPOUND_MARKERS = ("先别管", "先别贴", "先别整", "我就想知道", "其实主要", "等下你刚说")
SOFT_LEN = 85
EMOJI_RE = __import__("re").compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def score_turn(text: str) -> dict:
    leaks = [m for m in LEAK_MARKERS if m in text]
    compounds = [m for m in COMPOUND_MARKERS if m in text]
    return {
        "len": len(text),
        "long": len(text) > SOFT_LEN,
        "comma_heavy": text.count("，") >= 4,
        "leaks": leaks,
        "compounds": compounds,
        "compoundish": (
            len(compounds) >= 2
            or ("先别" in text and "想" in text)
            or ("我就想知道" in text and ("先别" in text or "其实" in text))
        ),
        "emoji": bool(EMOJI_RE.search(text)),
        "markdownish": ("**" in text) or ("```" in text) or text.count("\n\n") >= 1,
    }


def score_session(directory: Path) -> dict | None:
    persona_path = directory / "persona.json"
    chain_path = directory / "dialogue_chain.json"
    if not persona_path.is_file() or not chain_path.is_file():
        return None
    try:
        persona = json.loads(persona_path.read_text())
        chain = json.loads(chain_path.read_text())
    except (OSError, ValueError):
        return None
    turns = chain.get("turns") or []
    if not turns:
        return None
    scored = [score_turn(t["user"]["text"]) for t in turns]
    lens = [item["len"] for item in scored]
    return {
        "directory": str(directory),
        "persona_id": persona.get("id"),
        "prompt_version": persona.get("prompt_version") or "none",
        "turn_count": len(turns),
        "open_len": lens[0],
        "open_text": turns[0]["user"]["text"],
        "mean_len": round(mean(lens), 1),
        "max_len": max(lens),
        "long_turns": sum(1 for item in scored if item["long"]),
        "compound_turns": sum(1 for item in scored if item["compoundish"]),
        "leak_turns": sum(1 for item in scored if item["leaks"]),
        "emoji_turns": sum(1 for item in scored if item["emoji"]),
        "markdown_turns": sum(1 for item in scored if item["markdownish"]),
        "comma_heavy_turns": sum(1 for item in scored if item["comma_heavy"]),
    }


def summarize(constructed_root: Path) -> dict:
    sessions = []
    for path in sorted(constructed_root.iterdir()):
        if not path.is_dir():
            continue
        item = score_session(path)
        if item is not None:
            sessions.append(item)
    by_version = defaultdict(list)
    for item in sessions:
        by_version[item["prompt_version"]].append(item)

    def pack(items: list[dict]) -> dict:
        if not items:
            return {"sessions": 0}
        return {
            "sessions": len(items),
            "open_len_mean": round(mean(i["open_len"] for i in items), 1),
            "open_len_max": max(i["open_len"] for i in items),
            "utt_len_mean": round(mean(i["mean_len"] for i in items), 1),
            "utt_len_max": max(i["max_len"] for i in items),
            "sessions_with_long": sum(1 for i in items if i["long_turns"]),
            "sessions_with_compound": sum(1 for i in items if i["compound_turns"]),
            "sessions_with_leak": sum(1 for i in items if i["leak_turns"]),
            "sessions_with_emoji": sum(1 for i in items if i["emoji_turns"]),
            "sessions_with_markdown": sum(1 for i in items if i["markdown_turns"]),
        }

    return {
        "constructed_root": str(constructed_root),
        "total_scored": len(sessions),
        "by_version": {key: pack(value) for key, value in sorted(by_version.items())},
        "version_counts": dict(Counter(i["prompt_version"] for i in sessions)),
        "worst": sorted(
            sessions, key=lambda i: (i["max_len"], i["compound_turns"], i["leak_turns"]), reverse=True
        )[:8],
    }


def main():
    root = Path(__file__).resolve().parents[2]
    constructed = root / "data" / "paper_trail" / "constructed"
    report = summarize(constructed)
    out = root / "data" / "paper_trail" / "collections" / "quality_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["by_version"], ensure_ascii=False, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
