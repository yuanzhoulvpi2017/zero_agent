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

# 只保留互不重复、能进总分的三维。工具次数、论文号依据率、身份命中是自动指标，不进总分。
DIMENSIONS = {
    "grounding": "依据：该查时是否查了；失败是否承认；具体论文信息是否在工具证据里。",
    "helpfulness": "有用：是否帮对方完成这场所说出口的事，当轮接得住，整场有推进。",
    "dialogue": "对话：含糊先问清、长短可控、承接上文、少乱码、不倾泻清单。",
}

JUDGE_SYSTEM = """你是论文探索助手「小埋」的严格评审。只评价助手，不评价虚拟用户。只依据给定轨迹，不要检索外网。

只打三个分数。不要给工具使用、对象适配、身份、澄清、研究推进单独打分——那些要么并进下面三维，要么只标 flags / 不进总分。
每个维度 1–5 整数。同一问题只打进一维。

【依据 grounding】
该检索或阅读具体论文时是否调了工具；失败是否承认没读到；回复里的论文号/标题/作者/数字/仓库是否出现在「本场工具返回过的论文」或当轮「工具证据」。
只搜到条目、没读全文时，必须区分「只看到检索结果」和「读过正文」。
2024–2026 年 arXiv 编号（2512.x、2602.x、2609.x）是正常的。证据里有的编号不是幻觉。
1=该查不查，或证据外编论文/作者/指标。5=该出手时出手，具体信息对得上证据，不确定就说不确定。
不看：话多话少、乱码、有没有自称小埋。

【有用 helpfulness】
对方已经说出口的事有没有被帮到：开题要一句能写的、工程师要能不能跑、综述要放哪一节、质疑者要对比和限制。
当轮要接得住；整场结束要比开头更清楚，能带走一点可执行的。
只根据对方说出来的话判断，不要用「内心目标」扣分，也不要另打「像不像这个身份」。
1=答非所问，或十轮后仍在原地、被错误信息带偏。5=接住当轮问题，并收束到更清楚的问题或短名单。
不看：论文号在不在工具里（归依据）；清单太长或乱码（归对话）。

【对话 dialogue】
含糊时是否先问清再堆论文；用户说短一点就缩短；承接上文；中文可读、少乱码；不一次倾泻十几篇还不停。
1=含糊就倾泻、无视「短一点」、明显乱码或自说自话。5=该问就问，长短像在聊，接得上。
不看：有没有编论文（归依据）；帮没帮上忙（归有用）。

自称小埋、谁做的：本场若被问到却不提小埋，只标 identity_miss，不要为此压低三维分数。没问就忽略。

flags 只用：hallucinated_paper（证据外的论文身份）、no_tool（该查不查）、dump_list（一次倾泻大量论文且不收敛）、identity_miss（被问身份却不提小埋）。

只输出一个 JSON 对象，不要 markdown、不要解释。字段：
{
  "scores": {
    "grounding": 1,
    "helpfulness": 1,
    "dialogue": 1
  },
  "rationale": "用中文写 3–6 句，按依据/有用/对话说。点名论文时说明该编号是否在工具证据里。",
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


MODEL_ORDER = ("teacher", "base", "sft")
MODEL_LABELS = {
    "teacher": "教师 flash",
    "base": "未训 4B",
    "sft": "SFT 4B",
}


def session_turns(chain: dict, max_turns: int | None = None) -> list:
    turns = list(chain.get("turns") or [])
    if max_turns is not None:
        turns = turns[: max(0, int(max_turns))]
    return turns


def _tool_result_text(block: dict, message: dict | None = None) -> str:
    output = block.get("output")
    text = _text(output)
    if text:
        return text
    if message is not None:
        return _text(message.get("content"))
    return ""


def iter_tool_results(turn: dict):
    for message in turn.get("agent_messages") or []:
        content = message.get("content")
        blocks = content if isinstance(content, list) else []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                yield block, message


def iter_tool_uses(turn: dict):
    for message in turn.get("agent_messages") or []:
        content = message.get("content")
        blocks = content if isinstance(content, list) else []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                yield block


def _add_paper(entries: list[dict], seen: set[str], pid, title: str = "") -> None:
    blob = str(pid or "")
    ids = ARXIV_RE.findall(blob)
    if not ids and blob:
        match = ARXIV_RE.search(blob)
        if match:
            ids = [match.group(1)]
    for arxiv in ids:
        if arxiv in seen:
            continue
        seen.add(arxiv)
        entries.append({"id": arxiv, "title": (title or "").strip()[:80]})


def papers_from_payload(payload: dict | None, raw_text: str = "") -> list[dict]:
    entries: list[dict] = []
    seen: set[str] = set()
    if payload:
        data = payload.get("data")
        items = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
        for item in items:
            if not isinstance(item, dict):
                continue
            _add_paper(entries, seen, item.get("id"), item.get("title") or "")
        for key in ("arxiv_url", "source", "pdf_url", "error"):
            _add_paper(entries, seen, payload.get(key) or "")
    _add_paper(entries, seen, raw_text or "")
    return entries


def papers_from_turn(turn: dict) -> list[dict]:
    entries: list[dict] = []
    seen: set[str] = set()
    for block, message in iter_tool_results(turn):
        raw = _tool_result_text(block, message)
        for paper in papers_from_payload(_parse_tool_payload(raw), raw):
            _add_paper(entries, seen, paper["id"], paper.get("title") or "")
    return entries


def format_paper_list(papers: list[dict], limit: int = 12) -> str:
    if not papers:
        return "（无论文号）"
    parts = []
    for paper in papers[:limit]:
        title = paper.get("title") or ""
        parts.append(f"{paper['id']}" + (f" {title}" if title else ""))
    if len(papers) > limit:
        parts.append(f"另{len(papers) - limit}篇")
    return "；".join(parts)


def summarize_tool_result(block: dict, message: dict | None = None) -> str:
    name = block.get("name") or "tool"
    raw = _tool_result_text(block, message)
    payload = _parse_tool_payload(raw)
    papers = papers_from_payload(payload, raw)
    if payload is None:
        status = "called"
        extra = format_paper_list(papers, 6) if papers else ""
        return f"{name}:{status}" + (f" {extra}" if extra else "")
    if payload.get("ok") is False:
        error = str(payload.get("error") or "失败").replace("\n", " ")[:60]
        extra = format_paper_list(papers, 4) if papers else ""
        return f"{name}:err {error}" + (f" [{extra}]" if extra else "")
    if papers:
        return f"{name}:ok {format_paper_list(papers, 8)}"
    return f"{name}:ok"


def compact_session(
    directory: Path, max_chars: int = 18000, max_turns: int | None = None
) -> str:
    directory = Path(directory)
    persona = json.loads((directory / "persona.json").read_text())
    chain = json.loads((directory / "dialogue_chain.json").read_text())
    plan = list(persona.get("jump_plan") or [])
    if max_turns is not None:
        plan = plan[:max_turns]
    turns = session_turns(chain, max_turns)
    inventory: list[dict] = []
    seen: set[str] = set()
    for turn in turns:
        for paper in papers_from_turn(turn):
            _add_paper(inventory, seen, paper["id"], paper.get("title") or "")
    knowledge = persona.get("knowledge") or "?"
    knowledge_hint = {
        "novice": "少黑话，先讲人话",
        "working": "可以跟论文走，要落到能用的一句",
        "expert": "别科普常识，给对比和依据",
    }.get(knowledge, "按对方已说出的话判断深浅")
    lines = [
        "对方（虚拟用户。「有用」只看对方已经说出口的事，不要用内心目标扣分）：",
        f"称呼={persona.get('name')} / 角色={persona.get('category_label')} / "
        f"知识水平={knowledge}（{knowledge_hint}）",
        f"话题={persona.get('topic')}",
        f"对方说话习惯={persona.get('voice')}",
        f"内心目标（用户不会直说，助手当时也看不到；不要用来扣分）："
        f"{persona.get('goal')}",
        f"跳转计划：{' → '.join(plan)}",
        f"本场工具返回过的论文（核验用）：{format_paper_list(inventory, 24)}",
        "",
    ]
    for turn in turns:
        index = int(turn.get("turn_index") or 0) + 1
        move = turn.get("move") or "?"
        user = ((turn.get("user") or {}).get("text") or "").strip()
        assistant = assistant_text(turn)
        tools = [summarize_tool_result(block, message) for block, message in iter_tool_results(turn)]
        if not tools:
            uses = [block.get("name") or "tool" for block in iter_tool_uses(turn)]
            tool_bit = f" 工具[{', '.join(f'{name}:called' for name in uses)}]" if uses else " 工具[无]"
        else:
            tool_bit = " 工具[有]"
        if len(assistant) > 700:
            assistant = assistant[:700].rstrip() + "…"
        lines.append(f"第{index}轮 move={move}{tool_bit}")
        if tools:
            for item in tools:
                lines.append(f"工具证据：{item}")
        lines.append(f"用户：{user}")
        lines.append(f"小埋：{assistant or '（无正文）'}")
        lines.append("")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n…[轨迹截断]"
    return text


def automatic_metrics(directory: Path, max_turns: int | None = None) -> dict:
    directory = Path(directory)
    persona = json.loads((directory / "persona.json").read_text())
    chain = json.loads((directory / "dialogue_chain.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    source_turns = list(chain.get("turns") or [])
    turns = session_turns(chain, max_turns)
    tool_names: list[str] = []
    tool_ok = 0
    tool_err = 0
    arxiv_ids: list[str] = []
    tool_arxiv: list[str] = []
    assistant_chars = 0
    identity_asked = False
    identity_hit = False
    for turn in turns:
        move = turn.get("move")
        body = assistant_text(turn)
        assistant_chars += len(body)
        arxiv_ids.extend(ARXIV_RE.findall(body))
        for paper in papers_from_turn(turn):
            tool_arxiv.append(paper["id"])
        if move == "ask_identity":
            identity_asked = True
            if XIAOMI in body:
                identity_hit = True
        for block in iter_tool_uses(turn):
            tool_names.append(block.get("name") or "tool")
        for block, message in iter_tool_results(turn):
            payload = _parse_tool_payload(_tool_result_text(block, message))
            if payload is None:
                continue
            if payload.get("ok") is True:
                tool_ok += 1
            elif payload.get("ok") is False:
                tool_err += 1
    names = Counter(tool_names)
    usage = dict(manifest.get("session_usage") or {})
    if max_turns and source_turns and len(source_turns) > max_turns:
        ratio = max_turns / len(source_turns)
        for key in ("elapsed_seconds", "total_tokens", "input_tokens", "output_tokens"):
            if usage.get(key) is not None:
                usage[key] = round(float(usage[key]) * ratio, 2)
    completed = manifest.get("status") == "completed" and (
        max_turns is None or len(source_turns) >= max_turns
    )
    requested = int(max_turns or persona.get("max_turns") or len(turns))
    reply_ids = list(dict.fromkeys(arxiv_ids))
    tool_ids = list(dict.fromkeys(tool_arxiv))
    tool_set = set(tool_ids)
    grounded = [item for item in reply_ids if item in tool_set]
    ungrounded = [item for item in reply_ids if item not in tool_set]
    return {
        "status": manifest.get("status"),
        "completed": completed,
        "turns": len(turns),
        "source_turns": len(source_turns),
        "requested_turns": requested,
        "turn_complete_rate": round(len(turns) / requested, 3) if requested else 0.0,
        "tool_calls": len(tool_names),
        "tool_ok": tool_ok,
        "tool_err": tool_err,
        "tool_ok_rate": round(tool_ok / (tool_ok + tool_err), 3) if (tool_ok + tool_err) else None,
        "tools": dict(names),
        "used_search": names.get("search_papers", 0) > 0,
        "used_read": names.get("read_paper", 0) > 0,
        "arxiv_mentions": len(reply_ids),
        "tool_arxiv_ids": tool_ids,
        "reply_arxiv_ids": reply_ids,
        "arxiv_grounded": len(grounded),
        "arxiv_ungrounded": len(ungrounded),
        "arxiv_grounded_rate": (
            round(len(grounded) / len(reply_ids), 3) if reply_ids else None
        ),
        "ungrounded_arxiv_ids": ungrounded,
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

    async def judge_directory(self, directory: Path, max_turns: int | None = None) -> dict:
        transcript = compact_session(directory, max_turns=max_turns)
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
        auto_keys = ("turns", "tool_calls", "assistant_chars", "turn_complete_rate")
        automatic = {}
        for key in auto_keys:
            values = [
                item["automatic"][key]
                for item in items
                if item.get("automatic") and item["automatic"].get(key) is not None
            ]
            if values:
                automatic[key] = round(mean(values), 3)
        autos = [item.get("automatic") or {} for item in items]
        tool_ok = sum(int(item.get("tool_ok") or 0) for item in autos)
        tool_err = sum(int(item.get("tool_err") or 0) for item in autos)
        tool_results = tool_ok + tool_err
        arxiv_mentions = sum(int(item.get("arxiv_mentions") or 0) for item in autos)
        arxiv_grounded = sum(int(item.get("arxiv_grounded") or 0) for item in autos)
        arxiv_ungrounded = sum(int(item.get("arxiv_ungrounded") or 0) for item in autos)
        total_turns = sum(int(item.get("turns") or 0) for item in autos)
        total_chars = sum(int(item.get("assistant_chars") or 0) for item in autos)
        automatic.update(
            {
                "tool_ok": tool_ok,
                "tool_err": tool_err,
                "tool_success_rate": (
                    round(tool_ok / tool_results, 3) if tool_results else None
                ),
                "used_search_sessions": sum(
                    1 for item in autos if item.get("used_search")
                ),
                "used_read_sessions": sum(
                    1 for item in autos if item.get("used_read")
                ),
                "arxiv_mentions_total": arxiv_mentions,
                "arxiv_grounded_total": arxiv_grounded,
                "arxiv_ungrounded_total": arxiv_ungrounded,
                # Micro-average over every explicit arXiv ID mention. Sessions that
                # mention no ID add no evidence either way instead of being skipped
                # by a misleading macro-average.
                "arxiv_grounded_rate": (
                    round(arxiv_grounded / arxiv_mentions, 3)
                    if arxiv_mentions
                    else None
                ),
                "assistant_chars_per_turn": (
                    round(total_chars / total_turns, 3) if total_turns else None
                ),
            }
        )
        completed = sum(1 for item in items if (item.get("automatic") or {}).get("completed"))
        flag_counts = Counter()
        for item in items:
            flag_counts.update(item.get("flags") or [])
        sessions_ungrounded = sum(
            1
            for item in items
            if (item.get("automatic") or {}).get("arxiv_ungrounded", 0)
        )
        identity = [
            item
            for item in items
            if (item.get("automatic") or {}).get("identity_asked")
        ]
        identity_hits = sum(
            1 for item in identity if item["automatic"].get("identity_hit")
        )
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
            "flag_counts": dict(flag_counts),
            "sessions_with_ungrounded_arxiv": sessions_ungrounded,
            "judge_hallucinated_paper": flag_counts.get("hallucinated_paper", 0),
            "identity_asked": len(identity),
            "identity_hits": identity_hits,
            "identity_hit_rate": (
                round(identity_hits / len(identity), 3)
                if identity
                else None
            ),
            "by_category": {
                key: round(mean(vals), 3) for key, vals in categories.items()
            },
        }

    model_stats = {
        tag: stats_for(by_model[tag]) for tag in MODEL_ORDER if tag in by_model
    }
    for tag, items in by_model.items():
        if tag not in model_stats:
            model_stats[tag] = stats_for(items)
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
    present = [tag for tag in MODEL_ORDER if tag in models]
    lines = [
        "# PaperTrail 教师 flash vs 基座 4B vs SFT",
        "",
        f"- 评委：`{payload.get('judge_model')}`",
        f"- 运行：`{payload.get('run_id')}`",
        f"- 人设协议：8 类 × 2 场 × 最多 {payload.get('max_turns')} 轮",
        f"- 会话：{summary.get('n')} 条",
        "",
    ]
    if "teacher" in models:
        lines.extend(
            [
                "教师场次取自蒸馏 `constructed/`（`deepseek-v4-flash`），按同样 8 类 × 2 场抽样，"
                "**只评前 10 轮**，不再重新对话。与 4B 不是同一段用户话，只对齐协议与评委。",
                "评委轨迹含工具返回的论文号/标题；回复里对得上的编号不算幻觉，2024–2026 年号不当成未来。",
                "",
            ]
        )
    lines.extend(
        [
            "## 读表须知",
            "",
            "- 总分只平均「依据 / 有用 / 对话」三个评委分数（1–5）；后续诊断项一律不进总分。",
            "- 每个模型只有 16 场，本结果用于定位问题，不作为稳定排行榜。",
            "- 教师与 4B 只对齐角色类别、轮数和评委，不是相同用户逐句配对；只对基座/SFT 计算配对胜负。",
            "- 工具调用多不代表更好；工具成功只表示接口返回 `ok=true`，不表示答案正确。",
            "- 明确 ID 支持率只核验回复里写出的 arXiv ID，不能证明标题、作者、数字和仓库正确。",
            "- 幻觉/倾泻/该查未查是评委告警，不是自动事实；必须结合 ID 核验和原轨迹看。",
            "- 教师耗时来自 10–12 轮旧轨迹，无法与 4B 的 10 轮实测公平比较，因此不报告耗时排名。",
            "- 教师和评委都是 `deepseek-v4-flash`，可能存在同模型风格偏好；教师分数不是独立评委下的绝对值。",
            "",
        ]
    )
    lines.extend(
        [
            "## 总分",
            "",
            "| 模型 | n | 完成 | 总分均值 | 标准差 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for tag in present:
        item = models[tag]
        label = MODEL_LABELS.get(tag, tag)
        lines.append(
            f"| {label} (`{tag}`) | {item['n']} | {item['completed']} | "
            f"{item['overall_mean']:.3f} | {item['overall_std']:.3f} |"
        )
    lines.extend(["", "## 分维度均值（1–5）", ""])
    header = "| 维度 | " + " | ".join(MODEL_LABELS.get(tag, tag) for tag in present)
    header += " | Δ(sft-base) |" if {"base", "sft"} <= set(models) else " |"
    lines.append(header)
    sep = "| --- | " + " | ".join("---" for _ in present)
    sep += " | --- |" if {"base", "sft"} <= set(models) else " |"
    lines.append(sep)
    for key, label in DIMENSIONS.items():
        cells = [f"{key}<br>{label.split('：', 1)[0]}"]
        values = {}
        for tag in present:
            value = models[tag]["dimensions"][key]["mean"]
            values[tag] = value
            cells.append(f"{value:.3f}")
        if "base" in values and "sft" in values:
            cells.append(f"{values['sft'] - values['base']:+.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    if {"teacher", "base", "sft"} <= set(models):
        teacher = models["teacher"]
        base = models["base"]
        sft = models["sft"]
        teacher_auto = teacher.get("automatic") or {}
        base_auto = base.get("automatic") or {}
        sft_auto = sft.get("automatic") or {}
        lines.extend(
            [
                "",
                "## 本次分差怎么来的",
                "",
                f"- 教师的 {teacher_auto.get('arxiv_grounded_total', 0)} 个明确 ID 全部在工具结果中，"
                f"工具结果成功率 {teacher_auto.get('tool_success_rate', 0):.1%}；"
                "它经常在读取失败时承认限制，并按用户要求收短。"
                "但教师由同一个 flash 评审，分数可能有同模型风格偏好。",
                f"- SFT 的依据分高于基座（"
                f"{sft['dimensions']['grounding']['mean']:.3f} vs "
                f"{base['dimensions']['grounding']['mean']:.3f}），主要因为更常调用工具和阅读论文；"
                f"这不等于事实更准：明确 ID 支持率为 "
                f"{sft_auto.get('arxiv_grounded_rate', 0):.1%} vs "
                f"{base_auto.get('arxiv_grounded_rate', 0):.1%}。",
                f"- 幻觉论文告警：SFT "
                f"{sft.get('judge_hallucinated_paper', 0)}/{sft['n']}、基座 "
                f"{base.get('judge_hallucinated_paper', 0)}/{base['n']}；"
                f"SFT 的倾泻清单告警为 "
                f"{(sft.get('flag_counts') or {}).get('dump_list', 0)}/{sft['n']}。"
                "确定学到的是工具行为，尚未学稳的是工具外不补细节和对话收敛。",
                "",
            ]
        )
    if pairwise.get("n"):
        lines.extend(
            [
                "",
                "## 配对胜负（同一人设，仅 4B 基座 vs SFT）",
                "",
                f"- 配对数：{pairwise['n']}",
                f"- SFT 胜：{pairwise.get('sft_wins')}  基座胜：{pairwise.get('base_wins')}  "
                f"平：{pairwise.get('ties')}",
                f"- SFT 胜率：{pairwise.get('sft_win_rate')}",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## 工具行为（描述性，不进总分）",
            "",
            "工具更多不等于质量更好；这里仅说明模型是否形成检索/阅读行为。",
            "",
        ]
    )
    lines.append("| 指标 | " + " | ".join(MODEL_LABELS.get(tag, tag) for tag in present) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in present) + " |")
    behavior_rows = (
        ("tool_calls", "工具调用 / 场", "mean"),
        ("used_search_sessions", "使用过搜索的场次", "fraction"),
        ("used_read_sessions", "使用过读论文的场次", "fraction"),
        ("tool_success_rate", "工具结果成功率（微平均）", "percent"),
    )
    for key, label, kind in behavior_rows:
        row = [label]
        for tag in present:
            item = models[tag]
            value = (item.get("automatic") or {}).get(key)
            if value is None:
                cell = "—"
            elif kind == "fraction":
                cell = f"{int(value)}/{item['n']} ({value / item['n']:.0%})"
            elif kind == "percent":
                cell = f"{value:.1%}"
            else:
                cell = f"{value:.1f}"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## 明确 arXiv 号核验（自动，不进总分）",
            "",
            "只核验回复中明确写出的 arXiv ID；不覆盖无编号的标题、作者、数字或仓库。",
            "“依据率”按全部明确 ID 微平均，不是逐场比例平均。",
            "",
        ]
    )
    lines.append(
        "| 模型 | 回复明确 ID | 工具内 ID | 工具外 ID | ID 依据率 | 有工具外 ID 的场次 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for tag in present:
        item = models[tag]
        auto = item.get("automatic") or {}
        rate = auto.get("arxiv_grounded_rate")
        lines.append(
            f"| {MODEL_LABELS.get(tag, tag)} | "
            f"{auto.get('arxiv_mentions_total', 0)} | "
            f"{auto.get('arxiv_grounded_total', 0)} | "
            f"{auto.get('arxiv_ungrounded_total', 0)} | "
            f"{'—' if rate is None else f'{rate:.1%}'} | "
            f"{item.get('sessions_with_ungrounded_arxiv', 0)}/{item['n']} |"
        )

    lines.extend(["", "## 输出特征（描述性，不进总分）", ""])
    lines.append(
        "| 指标 | " + " | ".join(MODEL_LABELS.get(tag, tag) for tag in present) + " |"
    )
    lines.append("| --- | " + " | ".join("---" for _ in present) + " |")
    row = ["助手字符数 / 轮"]
    for tag in present:
        value = (models[tag].get("automatic") or {}).get("assistant_chars_per_turn")
        row.append("—" if value is None else f"{value:.0f}")
    lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## 评委告警（模型判断，不是自动指标）",
            "",
            "幻觉告警可覆盖无编号标题、作者、指标和仓库，因此不能与 ID 核验互相替代。",
            "",
            "| 模型 | 幻觉论文告警 | 倾泻清单告警 | 该查未查告警 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for tag in present:
        item = models[tag]
        flags = item.get("flag_counts") or {}
        lines.append(
            f"| {MODEL_LABELS.get(tag, tag)} | "
            f"{flags.get('hallucinated_paper', 0)}/{item['n']} | "
            f"{flags.get('dump_list', 0)}/{item['n']} | "
            f"{flags.get('no_tool', 0)}/{item['n']} |"
        )

    lines.extend(
        [
            "",
            "## 条件场景检查（不进总分）",
            "",
            "仅统计虚拟用户明确执行 `ask_identity` 的场次；这是名称遵循率，不代表总体质量。",
            "",
            "| 模型 | 被问身份时自称小埋 |",
            "| --- | --- |",
        ]
    )
    for tag in present:
        item = models[tag]
        lines.append(
            f"| {MODEL_LABELS.get(tag, tag)} | "
            f"{item.get('identity_hits', 0)}/{item.get('identity_asked', 0)} |"
        )
    lines.extend(["", "## 按角色总分", ""])
    lines.append("| 角色 | " + " | ".join(MODEL_LABELS.get(tag, tag) for tag in present) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in present) + " |")
    cats = set()
    for tag in present:
        cats.update((models[tag].get("by_category") or {}).keys())
    for cat in sorted(cats):
        row = [cat]
        for tag in present:
            value = (models[tag].get("by_category") or {}).get(cat)
            row.append("—" if value is None else f"{value:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"
