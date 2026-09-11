"""Dialogue chain projection for SFT-ready distillation records."""

from __future__ import annotations

MAX_CHAIN_TURNS = 20
CHAIN_SCHEMA = "paper_trail.dialogue_chain.v1"


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _has_thinking(content) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "thinking"
        for block in content
    )


def build_dialogue_chain(
    *,
    session_id: str,
    trajectory: list,
    llm_calls: list | None = None,
    persona: dict | None = None,
    moves: list | None = None,
    manifest: dict | None = None,
) -> dict:
    """One turn = one user ask + following assistant/tool messages until next user."""
    llm_calls = list(llm_calls or [])
    moves = list(moves or [])
    turns = []
    current = None
    for message in trajectory:
        role = message.get("role")
        if role == "user":
            if current is not None:
                turns.append(current)
            index = len(turns)
            current = {
                "turn_index": index,
                "move": moves[index] if index < len(moves) else None,
                "user": {
                    "message_id": message.get("id"),
                    "text": _text(message.get("content")),
                    "timestamp": message.get("timestamp"),
                },
                "assistant": {
                    "message_ids": [],
                    "texts": [],
                    "thinking_present": False,
                    "timestamp": None,
                },
                "agent_messages": [message],
                "llm_calls": [],
            }
            continue
        if current is None:
            continue
        current["agent_messages"].append(message)
        if role == "assistant":
            content = message.get("content")
            text = _text(content)
            current["assistant"]["message_ids"].append(message.get("id"))
            if text:
                current["assistant"]["texts"].append(text)
            if _has_thinking(content):
                current["assistant"]["thinking_present"] = True
            current["assistant"]["timestamp"] = message.get("timestamp")
    if current is not None:
        turns.append(current)

    if len(turns) > MAX_CHAIN_TURNS:
        turns = turns[:MAX_CHAIN_TURNS]

    for turn in turns:
        turn["assistant"]["text"] = "\n\n".join(turn["assistant"]["texts"]).strip()
        turn["assistant"].pop("texts", None)
        turn["llm_calls"] = [
            call for call in llm_calls if call.get("turn_index") == turn["turn_index"]
        ]

    return {
        "schema": CHAIN_SCHEMA,
        "session_id": session_id,
        "turn_definition": "用户一句 + 小埋一整轮回复（含中间工具）算 1 轮",
        "max_turns": MAX_CHAIN_TURNS,
        "turn_count": len(turns),
        "persona": persona,
        "manifest": {
            "status": (manifest or {}).get("status"),
            "model": ((manifest or {}).get("config") or {}).get("model"),
            "title": (manifest or {}).get("title"),
            "session_usage": (manifest or {}).get("session_usage"),
        },
        "records": {
            "trajectory": "trajectory.json",
            "llm_calls": "llm_calls.json",
            "conversation": "conversation.json",
            "dialogue_chain": "dialogue_chain.json",
        },
        "turns": turns,
    }
