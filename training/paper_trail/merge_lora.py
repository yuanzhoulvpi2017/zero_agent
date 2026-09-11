"""Merge PaperTrail QLoRA adapter into a bf16 checkpoint for vLLM."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=repo_root() / "model" / "Qwen" / "Qwen3.5-4B",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=repo_root()
        / "data"
        / "paper_trail"
        / "models"
        / "qwen35-4b-sft-qlora-16k"
        / "adapter",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root()
        / "data"
        / "paper_trail"
        / "models"
        / "qwen35-4b-sft-qlora-16k"
        / "merged",
    )
    args = parser.parse_args()
    if not args.base.is_dir():
        raise SystemExit(f"missing base model: {args.base}")
    if not args.adapter.is_dir():
        raise SystemExit(f"missing adapter: {args.adapter}")

    print(f"base={args.base}", flush=True)
    print(f"adapter={args.adapter}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter))
    merged = model.merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output), safe_serialization=True)
    tokenizer.save_pretrained(str(args.output))
    print(f"merged -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
