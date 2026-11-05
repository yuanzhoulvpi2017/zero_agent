"""一个入口：model logs -> 清洗后的 SFT -> 全量 loss mask 检查，不运行训练。

从任意目录运行，复用训练环境，仅加载本地 tokenizer：
    training/paper_trail/.venv/bin/python agent/paper_trail/simple/code002_to_sft.py

--input 指定 simple 总目录或一个会话目录；默认处理全部会话的全部调用。
--output-dir 必须尚不存在；相对路径以仓库根目录解析。
所有产物统一写入 data/paper_trail/mask_demo/<时间_UUID>/。
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4

import torch
import transformers
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
# 这是 simple 教学入口；复用 dataset/ 的格式规范化，不另造一套格式。
sys.path.insert(0, str(ROOT / "dataset"))
from paper_trail_data.to_sft import call_to_sample, write_jsonl


def resolve_path(path: Path) -> Path:
    return (ROOT / path).resolve()


def prepare_sample(sample: dict) -> dict:
    """必须携带原始工具 schema；只清理 assistant 的 reasoning，不动工具返回。"""
    sample = json.loads(json.dumps(sample, ensure_ascii=False))
    tools = sample.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("样本必须携带非空 tools schema；请从 model log 转换，不补造工具定义")
    names = set()
    for tool in tools:
        function = tool.get("function", {})
        if tool.get("type") != "function" or not function.get("name") or not isinstance(function.get("parameters"), dict):
            raise ValueError("tools 必须包含 function.name 和 function.parameters")
        names.add(function["name"])
    for message in sample["messages"]:
        if message["role"] != "assistant":
            continue
        for key in ("thinking", "reasoning_content", "reasoning", "reason", "reasoning_details"):
            message.pop(key, None)
        # 有工具调用就仅保留结构化动作；目标与历史使用同一清洗规则。
        if message.get("tool_calls"):
            message["content"] = ""
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("SFT assistant content 必须是字符串")
        # 兼容旧数据里直接写进正文开头的 thinking，避免重新带进训练文本。
        content = re.sub(r"^\s*<think>.*?</think>\s*", "", content, count=1, flags=re.DOTALL)
        if "<think>" in content or "</think>" in content:
            raise ValueError("assistant 正文残留未配对或非前缀 think 标签，请先清洗")
        message["content"] = content
        for call in message.get("tool_calls") or []:
            if call["function"]["name"] not in names:
                raise ValueError(f"工具调用没有对应 schema：{call['function']['name']}")
    return sample


def build_batch(tokenizer, sample: dict) -> tuple[dict, str]:
    sample = prepare_sample(sample)
    messages = sample["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("一次模型调用的 SFT 样本必须以本次 assistant 输出结尾")
    kwargs = {"tools": sample["tools"], "enable_thinking": False}
    # 1. 原始模型模板编码完整对话，包含真实工具 schema 和本次模型输出。
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt", **kwargs,
    )
    # 2. 同一原始模板编码 prompt：去掉本次输出，再加生成前缀。
    #    enable_thinking=False 让模板自行添加空思考包装，不拼接/替换模板字符串。
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", **kwargs,
    )
    ids = encoded["input_ids"]
    prefix_ids = prefix["input_ids"]
    prefix_length = prefix_ids.shape[1]
    if not torch.equal(ids[:, :prefix_length], prefix_ids):
        raise ValueError("原始模板的 prompt token 不是完整对话的精确前缀，无法安全计算 loss mask")
    if prefix_length >= ids.shape[1]:
        raise ValueError("本次 assistant 输出没有可监督的 token")
    attention = encoded["attention_mask"]

    # 3. 一个 model log 只监督末尾的本次输出，tool call 和正文一视同仁。
    #    历史 assistant 在它自己的调用样本里监督；此处作为上下文，避免重复监督。
    #    schema / 用户 / tool_result / 历史消息 / 原生生成前缀全部 mask=0。
    loss_mask = attention.bool().clone()
    loss_mask[:, :prefix_length] = False
    labels = ids.clone()
    labels[~loss_mask] = -100
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **kwargs,
    )
    # 4. causal shift：未来 logits[:, t] 预测 input_ids[:, t+1]。
    #    传给 Hugging Face causal LM 的 labels 保持原长度，模型内部负责移位。
    #    手算 CE 时使用 logits[:, :-1] 对 labels[:, 1:]，忽略 -100，
    #    最后除以有效目标数；不要将未移位的 mask 直接乘到移位后的 loss。
    return {"input_ids": ids, "attention_mask": attention,
            "loss_mask": loss_mask, "labels": labels}, rendered


def convert_log(record: dict, path: Path) -> dict | None:
    """一份日志 -> 一条样本；失败、被中断、空回复均不作为训练目标。"""
    if record.get("status") != "completed":
        return None

    # 原始日志结构：
    # request.messages = system / user / assistant / tool 的完整上下文
    # request.tools    = 工具名称、描述和参数 schema
    # response         = {text, thinking, tool_calls: [{id, name, input}]}
    #
    # call_to_sample 先规范化历史消息，再在末尾追加本次 assistant：
    # response.text -> messages[-1].content
    # response.tool_calls -> messages[-1].tool_calls
    #   {id, name, input} -> {id, type: "function", function: {name, arguments}}
    # input / arguments 统一为字典，供现有 Qwen chat template 渲染。
    # 历史工具返回保留 role="tool" 和 tool_call_id，与调用 id 对应。
    # request.tools -> 样本顶层 tools，不能丢掉，否则模型不知道工具定义。
    # include_thinking=False 同时去掉历史与目标中的 reasoning / thinking。
    sample = call_to_sample(
        record,
        session_id=path.parent.name,
        source=str(path),
        include_thinking=False,
        max_tokens=0,  # 教学示例保留完整内容，训练前再按目标 tokenizer 检查长度。
    )
    return prepare_sample(sample) if sample else None

def write_inspection(directory: Path, tokenizer, sample: dict, batch: dict, rendered: str) -> str:
    """每次调用一个检查目录，另有批次总索引，不再只检查一条样本。"""
    directory.mkdir(parents=True)
    ids, masks, labels = (batch[key][0].tolist() for key in ("input_ids", "loss_mask", "labels"))
    prompt = tokenizer.apply_chat_template(
        sample["messages"][:-1], tools=sample["tools"], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    supervised = tokenizer.decode([token for token, mask in zip(ids, masks, strict=True) if mask])
    segments = []
    start = 0
    for end in range(1, len(ids) + 1):
        if end == len(ids) or masks[end] != masks[start]:
            segments.append(f"[loss_mask={int(masks[start])} tokens={start}:{end}]\n" + tokenizer.decode(ids[start:end]))
            start = end
    segment_text = "\n\n".join(segments)
    for name, text in {"rendered.txt": rendered, "prompt.txt": prompt,
                       "supervised.txt": supervised, "mask_segments.txt": segment_text}.items():
        (directory / name).write_text(text, encoding="utf-8")
    for name, value in {"sample.json": sample, "batch.json": {key: value[0].tolist() for key, value in batch.items()}}.items():
        (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (directory / "tokens.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["position", "input_id", "token", "attention_mask", "loss_mask", "label", "predicted_by_logits_at"])
        for i, (token_id, mask, label) in enumerate(zip(ids, masks, labels, strict=True)):
            writer.writerow([i, token_id, tokenizer.convert_ids_to_tokens(token_id),
                             int(batch["attention_mask"][0, i]), int(mask), label, i - 1 if i else ""])
    return segment_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/paper_trail/simple"))
    parser.add_argument("--model", type=Path, default=Path("model/Qwen/Qwen3.5-4B"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    input_dir = resolve_path(args.input)
    paths = sorted(input_dir.rglob("turn_*_call_*.json")) if input_dir.is_dir() else []
    if not paths:
        parser.error(f"未找到模型调用日志：{input_dir}")
    now = datetime.now(timezone.utc)
    version = now.strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]
    output = resolve_path(args.output_dir) if args.output_dir else ROOT / "data/paper_trail/mask_demo" / version
    if output.exists():
        parser.error(f"输出目录已存在，不覆盖：{output}")

    # 不改 tokenizer.chat_template，不切换 TRL 模板，不加载模型权重。
    tokenizer = AutoTokenizer.from_pretrained(str(resolve_path(args.model)), local_files_only=True)
    original_template = tokenizer.chat_template
    entries, sources, skipped = [], [], Counter()
    for path in paths:
        try:
            raw = path.read_bytes()
            record = json.loads(raw)
            sample = convert_log(record, path)
            source = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
            if sample is None:
                reason = "not_completed" if record.get("status") != "completed" else "empty_request_or_response"
                source["skipped"] = reason
                skipped[reason] += 1
            else:
                # 先对全部样本完成前缀验证，任何错误都报路径，不静默漏掉。
                batch, rendered = build_batch(tokenizer, sample)
                sample["sample_id"] = f"{path.parent.name}/{path.stem}"
                entries.append((sample, batch, rendered))
            sources.append(source)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            parser.error(f"无法处理 {path}：{exc}")
    if not entries:
        parser.error(f"没有可用样本：{dict(skipped)}")
    if tokenizer.chat_template != original_template:
        raise RuntimeError("原始模型模板被意外修改")

    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "samples.jsonl", [sample for sample, _, _ in entries])
    (output / "original_template.jinja").write_text(original_template, encoding="utf-8")
    index, encoded_rows, combined, counts = [], [], [], Counter()
    for sample, batch, rendered in entries:
        relative = Path("sessions") / sample["sample_id"]
        segments = write_inspection(output / relative, tokenizer, sample, batch, rendered)
        kind = "tool_call" if sample["messages"][-1].get("tool_calls") else "reply"
        counts[kind] += 1
        index.append({"sample_id": sample["sample_id"], "source": sample["source"],
                      "session_id": sample["session_id"], "turn_index": sample["turn_index"],
                      "call_index": sample["call_index"], "target_type": kind,
                      "tokens": batch["input_ids"].numel(), "target_tokens": int(batch["loss_mask"].sum()),
                      "inspection": relative.as_posix()})
        encoded_rows.append({"sample_id": sample["sample_id"], "source": sample["source"],
                             **{key: value[0].tolist() for key, value in batch.items()}})
        combined.append(f"===== {sample['sample_id']} | {kind} =====\n{segments}")
    write_jsonl(output / "encoded.jsonl", encoded_rows)
    (output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "all_mask_segments.txt").write_text("\n\n".join(combined), encoding="utf-8")
    sessions = sorted({sample["session_id"] for sample, _, _ in entries})
    manifest = {
        "status": "completed", "created_at": now.isoformat(),
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None,
        "code_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()),
        "config": {"input": str(input_dir), "tokenizer": str(resolve_path(args.model)),
                   "supervision": "final_assistant_only", "mask_method": "exact_prompt_token_prefix",
                   "drop_tool_call_text_in_target_and_history": True, "include_thinking": False,
                   "enable_thinking": False, "template_modified": False, "truncation": False},
        "template_sha256": hashlib.sha256(original_template.encode()).hexdigest(),
        "sources": sources, "trajectory_batches": sessions, "skipped": dict(skipped),
        "dataset": {"version": output.name, "path": str(output / "samples.jsonl"), "sessions": len(sessions),
                    "samples": len(entries), "target_types": dict(counts), "encoded_path": str(output / "encoded.jsonl")},
        "inspection_count": len(index), "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "base_model": None, "training": None, "test_set": None, "evaluation": None,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 全量 SFT 与 loss mask 检查", "",
             f"读取 {len(paths)} 份日志，生成 {len(entries)} 条样本及检查目录，覆盖 {len(sessions)} 个会话；跳过 {sum(skipped.values())} 条。", "",
             "[samples.jsonl](samples.jsonl) 是清洗后的 SFT；[encoded.jsonl](encoded.jsonl) 保存对应 input_ids / attention_mask / loss_mask / labels。",
             "[all_mask_segments.txt](all_mask_segments.txt) 汇总全部样本；下表逐条查看。", "",
             "有 tool_calls 的 assistant 只保留结构化调用，目标与历史正文都清空；工具返回原样保留。",
             "使用未经修改的模型原始模板，逐 token 校验 prompt 前缀。只监督本次输出，schema / 历史 / tool_result / 原生空 think 前缀均为 0。",
             "原生空 think 包装包含换行，不改写成紧凑标签。生产训练须保留此处 labels；若重新使用 assistant_only_loss=True，将改变监督策略。", "",
             "| 会话 | turn / call | 目标 | 总 token / 监督 token | 完整 mask | 仅本次监督文本 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in index:
        folder = row["inspection"]
        lines.append(f"| {row['session_id']} | {row['turn_index']} / {row['call_index']} | {row['target_type']} | {row['tokens']} / {row['target_tokens']} | [mask]({folder}/mask_segments.txt) | [target]({folder}/supervised.txt) |")
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 自定义输出也更新统一检查入口；旧实验不删不覆盖，只明确停止作为示例。
    hub = ROOT / "data/paper_trail/mask_demo"
    hub.mkdir(parents=True, exist_ok=True)
    (hub / "README.md").write_text(
        f"# Loss mask 当前检查入口\n\n[当前全量结果]({(output / 'README.md').as_posix()})："
        f"{len(sessions)} 个会话、{len(entries)} 次模型调用，全部已生成 mask。\n\n"
        "入口已合并为 code002_to_sft.py。其他旧目录只保留历史，包含单条样本、TRL 模板或手工模板等旧实现，均不作为当前检查依据。"
        "旧目录中 think 的不同边界是旧算法不同造成的，不是当前批次的策略。\n", encoding="utf-8",
    )
    print(f"完成：{len(sessions)} 个会话，{len(entries)} 条 SFT，{len(index)} 份 mask；目标类型={dict(counts)}")
    print(f"全部结果索引：{output / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
