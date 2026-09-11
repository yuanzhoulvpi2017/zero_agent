"""Automatic metrics and DeepSeek-flash judging for PaperTrail eval dialogues."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

import openai

ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)
XIAOMI = "小埋"

DIMENSIONS = {
    "tool_use": "工具使用：该检索/阅读时是否真的调用工具；失败时是否说明限制而不是编造。",
    "grounding": "来源依据：论文身份、链接、作者是否像真实工具结果；是否区分已读/未读。",
    "helpfulness": "有用程度：是否对准用户这一轮真正在问的事，而不是自说自话。",
    "clarification": "澄清追问：兴趣含糊时是否先问清，而不是一次堆砌大量论文。",
    "persona_fit": "人设适配：是否按对方知识水平和口气说话，不过于说教或过于空泛。",
    "dialogue_quality": "对话质量：是否承接上文、长度合适、像在聊天而不是贴说明书。",
    "research_progress": "研究推进：多轮之后问题、对比或下一步是否比开头更清楚。",
    "identity": "身份与边界：被问名字/创建者时是否自称小埋；不保证新颖、不编实验结果。",
}

JUDGE_SYSTEM = """你是论文探索助手「小埋」的严格评审。根据完整对话轨迹，只评价助手表现，不要评价虚拟用户。

打分规则：
- 每个维度给 1–5 的整数：1 很差，2 较差，3 一般，4 良好，5 优秀。
- 没有工具、却像亲历其境地讲具体论文/实验数字，grounding 和 tool_use 必须压低。
- 用户含糊时若助手直接倾泻十来篇论文，clarification 应偏低。
- 被问「你是谁/叫什么」却不提小埋或乱认身份，identity 应偏低；本段对话若没问身份，identity 给 3 或 4 即可，不要因此打满分。
- 只依据给定轨迹，不要检索外网。

只输出一个 JSON 对象，不要 markdown、不要解释。字段：
{
  "scores": {
    "tool_use": 1,
    "grounding": 1,
    "helpfulness": 1,
    "clarification": 1,
    "persona_fit": 1,
    "dialogue_quality": 1,
    "research_progress": 1,
    "identity": 1
  },
  "rationale": "用中文写 3–6 句：优点、问题、是否像在幻觉论文。",
  "flags": ["可空的短标签，如 hallucinated_paper / no_tool / dump_list / identity_miss"]
}
"""


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            parts.append(block["text"])
        elif block.get("type") == "tool_result":
            output = block.get("output")
            if isinstance(output, str):
                parts.append(output)
            elif isinstance(output, list):
                for item in output:
                    if isinstance(item, dict) and item.get("text"):
                        parts.append(item["text"])
                    elif isinstance(item, str):
                        parts.append(item)
    return "".join(parts)


def _parse_tool_payload(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def assistant_text(turn: dict) -> str:
    assistant = turn.get("assistant") or {}
    text = (assistant.get("text") or "").strip()
    if text:
        return text
    return "\n".join(assistant.get("texts") or []).strip()


def compact_session(directory: Path, max_chars: int = 9000) -> str:
    directory = Path(directory)
    persona = json.loads((directory / "persona.json").read_text())
    chain = json.loads((directory / "dialogue_chain.json").read_text())
    lines = [
        f"人设：{persona.get('name')} / {persona.get('category_label')} / "
        f"知识={persona.get('knowledge')} / 话题={persona.get('topic')} / "
        f"口吻={persona.get('voice')}",
        f"内心目标（助手不应被直接告知）：{persona.get('goal')}",
        f"跳转计划：{' → '.join(persona.get('jump_plan') or [])}",
        "",
    ]
    for turn in chain.get("turns") or []:
        index = int(turn.get("turn_index") or 0) + 1
        move = turn.get("move") or "?"
        user = ((turn.get("user") or {}).get("text") or "").strip()
        assistant = assistant_text(turn)
        tools = []
        for message in turn.get("agent_messages") or []:
            content = message.get("content")
            blocks = content if isinstance(content, list) else []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    name = block.get("name") or "tool"
                    payload = None
                    for later in turn.get("agent_messages") or []:
                        later_blocks = later.get("content") if isinstance(later.get("content"), list) else []
                        for item in later_blocks:
                            if not isinstance(item, dict) or item.get("type") != "tool_result":
                                continue
                            if item.get("id") != block.get("id") and item.get("name") != name:
                                continue
                            payload = _parse_tool_payload(_text(item.get("output") or later.get("content")))
                            if payload is not None:
                                break
                        if payload is not None:
                            break
                    status = "ok" if payload and payload.get("ok") else (
                        "err" if payload and payload.get("ok") is False else "called"
                    )
                    tools.append(f"{name}:{status}")
                elif block.get("type") == "tool_result":
                    continue
        tool_bit = f" 工具[{', '.join(tools)}]" if tools else " 工具[无]"
        if len(assistant) > 700:
            assistant = assistant[:700].rstrip() + "…"
        lines.append(f"第{index}轮 move={move}{tool_bit}")
        lines.append(f"用户：{user}")
        lines.append(f"小埋：{assistant or '（无正文）'}")
        lines.append("")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n…[轨迹截断]"
    return text


def automatic_metrics(directory: Path) -> dict:
    directory = Path(directory)
    persona = json.loads((directory / "persona.json").read_text())
    chain = json.loads((directory / "dialogue_chain.json").read_text())
    trajectory = json.loads((directory / "trajectory.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    turns = chain.get("turns") or []
    tool_names: list[str] = []
    tool_ok = 0
    tool_err = 0
    arxiv_ids: list[str] = []
    assistant_chars = 0
    identity_asked = False
    identity_hit = False
    for turn in turns:
        move = turn.get("move")
        body = assistant_text(turn)
        assistant_chars += len(body)
        arxiv_ids.extend(ARXIV_RE.findall(body))
        if move == "ask_identity":
            identity_asked = True
            if XIAOMI in body:
                identity_hit = True
        for message in turn.get("agent_messages") or []:
            content = message.get("content")
            blocks = content if isinstance(content, list) else []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tool_names.append(block.get("name") or "tool")
                elif block.get("type") == "tool_result":
                    payload = _parse_tool_payload(_text(block.get("output") or content))
                    if payload is None:
                        continue
                    if payload.get("ok") is True:
                        tool_ok += 1
                    elif payload.get("ok") is False:
                        tool_err += 1
    names = Counter(tool_names)
    usage = manifest.get("session_usage") or {}
    completed = manifest.get("status") == "completed"
    requested = int(persona.get("max_turns") or len(turns))
    return {
        "status": manifest.get("status"),
        "completed": completed,
        "turns": len(turns),
        "requested_turns": requested,
        "turn_complete_rate": round(len(turns) / requested, 3) if requested else 0.0,
        "tool_calls": len(tool_names),
        "tool_ok": tool_ok,
        "tool_err": tool_err,
        "tool_ok_rate": round(tool_ok / (tool_ok + tool_err), 3) if (tool_ok + tool_err) else None,
        "tools": dict(names),
        "used_search": names.get("search_papers", 0) > 0,
        "used_read": names.get("read_paper", 0) > 0,
        "arxiv_mentions": len(set(arxiv_ids)),
        "assistant_chars": assistant_chars,
        "elapsed_seconds": usage.get("elapsed_seconds"),
        "total_tokens": usage.get("total_tokens"),
        "model_calls": usage.get("model_calls"),
        "identity_asked": identity_asked,
        "identity_hit": identity_hit,
        "no_tool_but_arxiv": (not tool_names) and bool(arxiv_ids),
    }


def parse_json_object(text: str) -> dict:
    text = (text or "").strip()
    text = JSON_FENCE_RE.sub("", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge 未返回 JSON 对象")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("judge JSON 不是对象")
    return data


def clamp_score(value) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = 3
    return max(1, min(5, number))


def normalize_judgment(raw: dict) -> dict:
    scores_in = raw.get("scores") if isinstance(raw.get("scores"), dict) else {}
    scores = {key: clamp_score(scores_in.get(key, 3)) for key in DIMENSIONS}
    overall = round(mean(scores.values()), 3)
    flags = raw.get("flags") if isinstance(raw.get("flags"), list) else []
    flags = [str(item) for item in flags if item]
    return {
        "scores": scores,
        "overall": overall,
        "rationale": str(raw.get("rationale") or "").strip(),
        "flags": flags,
    }


class FlashJudge:
    def __init__(self, config: dict):
        self.config = config
        self.client = openai.AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=config["base_url"],
            timeout=min(120, int(config.get("request_timeout", 180))),
            max_retries=2,
        )
        self.model = config["model"]

    async def close(self):
        await self.client.close()

    async def _complete(self, messages: list, *, max_tokens: int = 900) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return (response.choices[0].message.content or "").strip()

    async def judge_directory(self, directory: Path) -> dict:
        transcript = compact_session(directory)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": "请评审下面这场对话。\n\n" + transcript,
            },
        ]
        text = await self._complete(messages)
        try:
            raw = parse_json_object(text)
        except (ValueError, json.JSONDecodeError):
            retry = messages + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": "刚才不是合法 JSON。请只重新输出那个 JSON 对象，不要其它文字。",
                },
            ]
            text = await self._complete(retry, max_tokens=700)
            raw = parse_json_object(text)
        judged = normalize_judgment(raw)
        judged["transcript_chars"] = len(transcript)
        return judged


def summarize_scores(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "models": {}}
    by_model: dict[str, list] = {}
    for row in rows:
        by_model.setdefault(row["model_tag"], []).append(row)

    def stats_for(items: list[dict]) -> dict:
        overalls = [item["overall"] for item in items]
        dim_means = {}
        for key in DIMENSIONS:
            values = [item["scores"][key] for item in items]
            dim_means[key] = {
                "mean": round(mean(values), 3),
                "std": round(pstdev(values), 3) if len(values) > 1 else 0.0,
            }
        auto_keys = (
            "turns",
            "tool_calls",
            "tool_ok_rate",
            "arxiv_mentions",
            "assistant_chars",
            "elapsed_seconds",
            "turn_complete_rate",
        )
        automatic = {}
        for key in auto_keys:
            values = [
                item["automatic"][key]
                for item in items
                if item.get("automatic") and item["automatic"].get(key) is not None
            ]
            if values:
                automatic[key] = round(mean(values), 3)
        completed = sum(1 for item in items if (item.get("automatic") or {}).get("completed"))
        identity = [
            item
            for item in items
            if (item.get("automatic") or {}).get("identity_asked")
        ]
        categories = {}
        for item in items:
            cat = item.get("category") or "unknown"
            categories.setdefault(cat, []).append(item["overall"])
        return {
            "n": len(items),
            "completed": completed,
            "overall_mean": round(mean(overalls), 3),
            "overall_std": round(pstdev(overalls), 3) if len(overalls) > 1 else 0.0,
            "dimensions": dim_means,
            "automatic": automatic,
            "identity_hit_rate": (
                round(
                    mean(
                        1.0 if item["automatic"].get("identity_hit") else 0.0
                        for item in identity
                    ),
                    3,
                )
                if identity
                else None
            ),
            "by_category": {
                key: round(mean(vals), 3) for key, vals in categories.items()
            },
        }

    model_stats = {tag: stats_for(items) for tag, items in by_model.items()}
    pairwise = {"n": 0, "sft_wins": 0, "base_wins": 0, "ties": 0, "dimension_wins": {}}
    by_persona: dict[str, dict] = {}
    for row in rows:
        by_persona.setdefault(row["persona_id"], {})[row["model_tag"]] = row
    dim_wins = {key: {"sft": 0, "base": 0, "tie": 0} for key in DIMENSIONS}
    for persona_id, pair in by_persona.items():
        if "sft" not in pair or "base" not in pair:
            continue
        pairwise["n"] += 1
        sft_o = pair["sft"]["overall"]
        base_o = pair["base"]["overall"]
        if sft_o > base_o:
            pairwise["sft_wins"] += 1
        elif base_o > sft_o:
            pairwise["base_wins"] += 1
        else:
            pairwise["ties"] += 1
        for key in DIMENSIONS:
            s_val = pair["sft"]["scores"][key]
            b_val = pair["base"]["scores"][key]
            if s_val > b_val:
                dim_wins[key]["sft"] += 1
            elif b_val > s_val:
                dim_wins[key]["base"] += 1
            else:
                dim_wins[key]["tie"] += 1
    pairwise["dimension_wins"] = dim_wins
    if pairwise["n"]:
        pairwise["sft_win_rate"] = round(pairwise["sft_wins"] / pairwise["n"], 3)
    return {"n": len(rows), "models": model_stats, "pairwise": pairwise}


def render_report(payload: dict) -> str:
    summary = payload.get("summary") or {}
    models = summary.get("models") or {}
    pairwise = summary.get("pairwise") or {}
    lines = [
        "# PaperTrail 4B 基座 vs SFT 对比",
        "",
        f"- 评委：`{payload.get('judge_model')}`",
        f"- 运行：`{payload.get('run_id')}`",
        f"- 人设：8 类 × 2 场 × 最多 {payload.get('max_turns')} 轮",
        f"- 会话：{summary.get('n')} 条（每模型应各 16）",
        "",
        "## 总分",
        "",
        "| 模型 | n | 完成 | 总分均值 | 标准差 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for tag in ("base", "sft"):
        item = models.get(tag)
        if not item:
            continue
        lines.append(
            f"| {tag} | {item['n']} | {item['completed']} | "
            f"{item['overall_mean']:.3f} | {item['overall_std']:.3f} |"
        )
    lines.extend(["", "## 分维度均值（1–5）", ""])
    header = "| 维度 | " + " | ".join(tag for tag in ("base", "sft") if tag in models) + " | Δ(sft-base) |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---" for _ in ("base", "sft") if _ in models) + " | --- |")
    for key, label in DIMENSIONS.items():
        cells = [f"{key}<br>{label.split('：', 1)[0]}"]
        values = {}
        for tag in ("base", "sft"):
            if tag not in models:
                continue
            value = models[tag]["dimensions"][key]["mean"]
            values[tag] = value
            cells.append(f"{value:.3f}")
        delta = ""
        if "base" in values and "sft" in values:
            delta = f"{values['sft'] - values['base']:+.3f}"
        cells.append(delta)
        lines.append("| " + " | ".join(cells) + " |")
    if pairwise.get("n"):
        lines.extend(
            [
                "",
                "## 配对胜负（同一人设）",
                "",
                f"- 配对数：{pairwise['n']}",
                f"- SFT 胜：{pairwise.get('sft_wins')}  基座胜：{pairwise.get('base_wins')}  "
                f"平：{pairwise.get('ties')}",
                f"- SFT 胜率：{pairwise.get('sft_win_rate')}",
                "",
            ]
        )
    lines.extend(["", "## 自动指标均值", ""])
    lines.append("| 指标 | base | sft |")
    lines.append("| --- | --- | --- |")
    auto_keys = (
        "turns",
        "tool_calls",
        "tool_ok_rate",
        "arxiv_mentions",
        "assistant_chars",
        "elapsed_seconds",
        "turn_complete_rate",
    )
    labels = {
        "turns": "完成轮数",
        "tool_calls": "工具调用次数",
        "tool_ok_rate": "工具成功比例",
        "arxiv_mentions": "提到的 arXiv 数",
        "assistant_chars": "助手字数",
        "elapsed_seconds": "耗时（秒）",
        "turn_complete_rate": "轮次完成率",
    }
    for key in auto_keys:
        row = [labels[key]]
        for tag in ("base", "sft"):
            if tag not in models:
                row.append("—")
                continue
            value = (models[tag].get("automatic") or {}).get(key)
            row.append("—" if value is None else f"{value:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", "## 按角色总分", ""])
    lines.append("| 角色 | base | sft |")
    lines.append("| --- | --- | --- |")
    cats = set()
    for tag in models:
        cats.update((models[tag].get("by_category") or {}).keys())
    for cat in sorted(cats):
        row = [cat]
        for tag in ("base", "sft"):
            value = (models.get(tag, {}).get("by_category") or {}).get(cat)
            row.append("—" if value is None else f"{value:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"
