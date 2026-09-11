"""Convert paper_trail constructed llm_calls into Qwen-style SFT JSONL.

Design notes
------------
- One sample = one teacher llm_call: full request.messages history + target assistant.
  That preserves before/after tool-call context (not a single isolated utterance).
- Strip DeepSeek-Flash thinking fields (response.thinking / message.reasoning_content /
  content blocks type=thinking / reason / reasoning). Default: no thinking in SFT.
- Never mid-cut the target assistant. If over budget: shrink oldest tool payloads first,
  then drop oldest dialogue turns as whole units (user + following assistant/tool).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
CONSTRUCTED_DIR = re.compile(
    r"^constructed__(?P<persona>[a-zA-Z0-9_-]+)__(?P<session>[0-9a-f]{32})$"
)

# Teacher (deepseek-v4-flash) thinking / reason fields to drop by default.
THINKING_MESSAGE_KEYS = (
    "reasoning_content",
    "thinking",
    "reason",
    "reasoning",
    "reasoning_details",
)
THINKING_RESPONSE_KEYS = (
    "thinking",
    "reasoning_content",
    "reason",
    "reasoning",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def constructed_root(root: Path | None = None) -> Path:
    return (root or repo_root()) / "data" / "paper_trail" / "constructed"


def content_to_text(content: Any, *, keep_thinking_blocks: bool = False) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                kind = item.get("type")
                if kind == "thinking" or "thinking" in item and kind not in {
                    "text",
                    "tool_use",
                    "tool_result",
                }:
                    if keep_thinking_blocks:
                        parts.append(str(item.get("thinking") or ""))
                    continue
                if kind == "text" or "text" in item:
                    parts.append(str(item.get("text") or ""))
                elif kind in {"tool_use", "tool_result"}:
                    parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    # Unknown structured block: keep JSON, but never thinking-only.
                    if any(k in item for k in THINKING_MESSAGE_KEYS) and "text" not in item:
                        continue
                    parts.append(json.dumps(item, ensure_ascii=False))
        return "".join(parts)
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "")
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def parse_arguments(arguments: Any) -> dict:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return {"_raw": arguments}
        return loaded if isinstance(loaded, dict) else {"_value": loaded}
    return {"_value": arguments}


def normalize_tool_calls(tool_calls: list | None) -> list[dict]:
    out: list[dict] = []
    for item in tool_calls or []:
        if not isinstance(item, dict):
            continue
        if "function" in item:
            function = item.get("function") or {}
            name = function.get("name") or item.get("name")
            arguments = parse_arguments(function.get("arguments"))
            call_id = item.get("id") or ""
        else:
            name = item.get("name")
            arguments = parse_arguments(item.get("input") or item.get("arguments"))
            call_id = item.get("id") or ""
        if not name:
            continue
        out.append(
            {
                "id": call_id or f"call_{len(out)}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return out


def soft_truncate(text: str, max_chars: int) -> str:
    """Truncate tool payload only; keeps a clear marker. Not used on assistant targets."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max(0, max_chars - 48)
    return text[:keep] + "\n…[tool_truncated]"


def strip_thinking_fields(message: dict) -> dict:
    cleaned = {
        key: value
        for key, value in message.items()
        if key not in THINKING_MESSAGE_KEYS
    }
    return cleaned


def normalize_message(
    message: dict,
    *,
    include_thinking: bool,
) -> dict | None:
    role = message.get("role")
    if role not in {"system", "user", "assistant", "tool"}:
        return None
    out: dict[str, Any] = {"role": role}
    if role == "tool":
        out["content"] = content_to_text(
            message.get("content"), keep_thinking_blocks=False
        )
        if message.get("tool_call_id"):
            out["tool_call_id"] = message["tool_call_id"]
        if message.get("name"):
            out["name"] = message["name"]
        return out

    text = content_to_text(
        message.get("content"), keep_thinking_blocks=include_thinking
    )
    tool_calls = normalize_tool_calls(message.get("tool_calls"))

    if role == "assistant":
        out["content"] = text
        if include_thinking:
            reasoning = None
            for key in THINKING_MESSAGE_KEYS:
                value = message.get(key)
                if isinstance(value, str) and value.strip():
                    reasoning = value
                    break
            if reasoning is None and isinstance(message.get("content"), list):
                thoughts = []
                for item in message["content"]:
                    if isinstance(item, dict) and (
                        item.get("type") == "thinking" or "thinking" in item
                    ):
                        thoughts.append(str(item.get("thinking") or ""))
                if thoughts:
                    reasoning = "".join(thoughts)
            if isinstance(reasoning, str) and reasoning.strip():
                out["reasoning_content"] = reasoning
        if tool_calls:
            out["tool_calls"] = tool_calls
        # Drop empty assistant turns with neither text nor tools.
        if not str(out.get("content") or "").strip() and not out.get("tool_calls"):
            return None
        return out

    out["content"] = text
    return out


def assistant_from_response(
    response: dict | None, *, include_thinking: bool
) -> dict | None:
    if not isinstance(response, dict):
        return None
    text = response.get("text") or ""
    tool_calls = normalize_tool_calls(response.get("tool_calls"))
    thinking = ""
    if include_thinking:
        for key in THINKING_RESPONSE_KEYS:
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                thinking = value
                break
    if not str(text).strip() and not tool_calls:
        # Pure-thinking teacher replies are useless once thinking is stripped.
        return None
    out: dict[str, Any] = {"role": "assistant", "content": text or ""}
    if include_thinking and thinking.strip():
        out["reasoning_content"] = thinking
    if tool_calls:
        out["tool_calls"] = tool_calls
    return strip_thinking_fields(out) if not include_thinking else out


def normalize_tools(tools: list | None) -> list[dict]:
    out: list[dict] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            function = dict(tool["function"])
            if function.get("parameters") is None:
                function["parameters"] = {"type": "object", "properties": {}}
            out.append({"type": "function", "function": function})
        elif "name" in tool:
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description") or "",
                        "parameters": tool.get("parameters")
                        or {"type": "object", "properties": {}},
                    },
                }
            )
    return out


def approx_message_chars(messages: list[dict], tools: list[dict] | None = None) -> int:
    total = 0
    if tools:
        total += len(json.dumps(tools, ensure_ascii=False))
    for message in messages:
        total += len(str(message.get("content") or ""))
        if message.get("tool_calls"):
            total += len(json.dumps(message["tool_calls"], ensure_ascii=False))
        for key in THINKING_MESSAGE_KEYS:
            value = message.get(key)
            if isinstance(value, str):
                total += len(value)
    return total


def render_length(
    messages: list[dict],
    tools: list[dict] | None,
    tokenizer,
    *,
    enable_thinking: bool,
) -> int:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": False,
        "tools": tools or None,
    }
    try:
        text = tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _turn_boundaries(messages: list[dict]) -> list[tuple[int, int]]:
    """Return [start, end) index spans for dialogue turns after the leading system."""
    spans: list[tuple[int, int]] = []
    start = None
    for index, message in enumerate(messages):
        if message.get("role") == "system" and index == 0:
            continue
        if message.get("role") == "user":
            if start is not None:
                spans.append((start, index))
            start = index
    if start is not None:
        spans.append((start, len(messages)))
    return spans


def fit_messages_to_budget(
    messages: list[dict],
    tools: list[dict] | None,
    *,
    max_tokens: int,
    tokenizer=None,
    enable_thinking: bool = False,
    min_tool_chars: int = 512,
    chars_per_token: float = 1.6,
) -> list[dict] | None:
    """Shrink context without cutting the final assistant target mid-content.

    Uses a fast character budget first. Optional tokenizer does at most one exact
    verification at the end (no per-step tokenize loops).
    """
    if not messages or messages[-1].get("role") != "assistant":
        return None
    fitted = [dict(message) for message in messages]
    char_budget = int(max_tokens * chars_per_token)

    def over_chars() -> bool:
        return approx_message_chars(fitted, tools) > char_budget

    if not over_chars():
        return fitted

    # 1) Prefer compressing oldest tool payloads (never the target assistant).
    guard = 0
    while over_chars() and guard < 64:
        guard += 1
        changed = False
        for message in fitted[:-1]:
            if message.get("role") != "tool":
                continue
            content = str(message.get("content") or "")
            if len(content) <= min_tool_chars:
                continue
            message["content"] = soft_truncate(
                content, max(min_tool_chars, len(content) // 2)
            )
            changed = True
            if not over_chars():
                break
        if not changed:
            break

    # 2) Drop oldest full turns (user + following assistant/tool), keep latest context.
    guard = 0
    while over_chars() and guard < 64:
        guard += 1
        spans = _turn_boundaries(fitted)
        if len(spans) <= 1:
            break
        drop_start, drop_end = spans[0]
        del fitted[drop_start:drop_end]
        if messages[0].get("role") == "system" and (
            not fitted or fitted[0].get("role") != "system"
        ):
            fitted.insert(0, dict(messages[0]))

    if over_chars():
        return None

    if tokenizer is not None:
        try:
            n_tokens = render_length(
                fitted, tools, tokenizer, enable_thinking=enable_thinking
            )
        except Exception:
            return fitted
        if n_tokens > max_tokens:
            # One more aggressive tool shrink pass, then re-check once.
            for message in fitted[:-1]:
                if message.get("role") == "tool":
                    content = str(message.get("content") or "")
                    if len(content) > min_tool_chars:
                        message["content"] = soft_truncate(content, min_tool_chars)
            try:
                n_tokens = render_length(
                    fitted, tools, tokenizer, enable_thinking=enable_thinking
                )
            except Exception:
                return fitted
            if n_tokens > max_tokens:
                return None
    return fitted


def call_to_sample(
    call: dict,
    *,
    session_id: str,
    source: str,
    include_thinking: bool,
    tokenizer=None,
    max_tokens: int = 0,
) -> dict | None:
    request = call.get("request") or {}
    messages_in = request.get("messages") or []
    if not isinstance(messages_in, list) or not messages_in:
        return None
    messages: list[dict] = []
    for message in messages_in:
        if not isinstance(message, dict):
            continue
        normalized = normalize_message(message, include_thinking=include_thinking)
        if normalized is None:
            continue
        if not include_thinking:
            normalized = strip_thinking_fields(normalized)
        messages.append(normalized)
    target = assistant_from_response(
        call.get("response"), include_thinking=include_thinking
    )
    if target is None:
        return None
    if not include_thinking:
        target = strip_thinking_fields(target)
    messages.append(target)
    tools = normalize_tools(request.get("tools"))

    if max_tokens > 0:
        approx_chars = approx_message_chars(messages, tools)
        # Fast path: only shrink when clearly large (char proxy for tokens).
        if approx_chars > int(max_tokens * 1.6):
            fitted = fit_messages_to_budget(
                messages,
                tools,
                max_tokens=max_tokens,
                tokenizer=None,
                enable_thinking=include_thinking,
            )
            if fitted is None:
                return None
            messages = fitted

    sample = {
        "session_id": session_id,
        "call_index": call.get("call_index"),
        "turn_index": call.get("turn_index"),
        "source": source,
        "teacher_model": call.get("model"),
        "messages": messages,
        "tools": tools,
        # Ensure Qwen template does not revive teacher thinking.
        "chat_template_kwargs": {"enable_thinking": bool(include_thinking)},
    }
    return sample


def iter_session_dirs(root: Path):
    if not root.is_dir():
        return
    for path in sorted(root.iterdir()):
        if path.is_dir() and CONSTRUCTED_DIR.fullmatch(path.name):
            yield path


def convert_session(
    directory: Path,
    *,
    include_thinking: bool,
    prompt_versions: set[str] | None,
    tokenizer=None,
    max_tokens: int = 0,
) -> list[dict]:
    persona_path = directory / "persona.json"
    calls_path = directory / "llm_calls.json"
    if not calls_path.is_file():
        return []
    session_id = directory.name
    match = CONSTRUCTED_DIR.fullmatch(directory.name)
    if match:
        session_id = match.group("session")
    if persona_path.is_file() and prompt_versions is not None:
        try:
            persona = json.loads(persona_path.read_text())
        except (OSError, ValueError):
            return []
        version = persona.get("prompt_version") or "none"
        if version not in prompt_versions:
            return []
    try:
        calls = json.loads(calls_path.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(calls, list):
        return []
    source = str(directory.relative_to(repo_root()))
    samples = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        sample = call_to_sample(
            call,
            session_id=session_id,
            source=source,
            include_thinking=include_thinking,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )
        if sample:
            samples.append(sample)
    return samples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--include-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep teacher thinking/reasoning (default: strip)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=65536,
        help="fit each sample under this token budget without cutting target assistant",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="tokenizer path used for budget fitting / preview",
    )
    parser.add_argument("--prompt-version", action="append", default=None)
    parser.add_argument("--limit-sessions", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--preview-template", action="store_true")
    args = parser.parse_args(argv)

    root = repo_root()
    input_root = args.input or constructed_root(root)
    output = args.output or (
        root / "data" / "paper_trail" / "datasets" / "sft" / "train.jsonl"
    )
    versions = set(args.prompt_version) if args.prompt_version else None
    model_path = args.model or (root / "model" / "Qwen" / "Qwen3.5-4B")

    tokenizer = None
    if args.preview_template:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    rows: list[dict] = []
    sessions = 0
    skipped = 0
    for directory in iter_session_dirs(input_root):
        samples = convert_session(
            directory,
            include_thinking=args.include_thinking,
            prompt_versions=versions,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
        )
        calls_path = directory / "llm_calls.json"
        try:
            n_calls = len(json.loads(calls_path.read_text())) if calls_path.is_file() else 0
        except (OSError, ValueError):
            n_calls = 0
        skipped += max(0, n_calls - len(samples))
        if not samples:
            continue
        sessions += 1
        rows.extend(samples)
        if sessions % 20 == 0:
            print(
                f"progress sessions={sessions} samples={len(rows)} skipped≈{skipped}",
                flush=True,
            )
        if args.limit_sessions and sessions >= args.limit_sessions:
            break
        if args.limit_samples and len(rows) >= args.limit_samples:
            rows = rows[: args.limit_samples]
            break

    write_jsonl(output, rows)
    manifest = {
        "input": str(input_root),
        "output": str(output),
        "sessions": sessions,
        "samples": len(rows),
        "skipped_over_budget_or_empty": skipped,
        "include_thinking": args.include_thinking,
        "max_tokens": args.max_tokens,
        "prompt_versions": sorted(versions) if versions else None,
        "mask_note": "TRL assistant_only_loss=True; thinking stripped unless --include-thinking.",
        "context_note": "Each sample keeps full llm_call history; over-budget shrinks oldest tools/turns, never the target assistant.",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    if args.preview_template and rows:
        sample = rows[0]
        try:
            text = tokenizer.apply_chat_template(
                sample["messages"],
                tools=sample.get("tools") or None,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=bool(args.include_thinking),
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                sample["messages"],
                tools=sample.get("tools") or None,
                tokenize=False,
                add_generation_prompt=False,
            )
        preview_path = output.with_suffix(".preview.txt")
        preview_path.write_text(text)
        print(f"preview_chars={len(text)} tokens≈{len(tokenizer.encode(text))} -> {preview_path}")
        for marker in ("<think>", "reasoning_content", "<tools>", "<tool_call>", "<tool_response>"):
            print(f"  contains {marker}: {marker in text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
