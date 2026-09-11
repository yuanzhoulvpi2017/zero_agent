"""TRL SFT for PaperTrail agent trajectories on Qwen3.5-4B."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def resolve_path(value: Path, root: Path, *, kind: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    root_candidate = (root / path).resolve()
    if kind == "file":
        return cwd_candidate if cwd_candidate.is_file() else root_candidate
    if kind == "dir_existing":
        return cwd_candidate if cwd_candidate.is_dir() else root_candidate
    # Output dirs: always anchor at repo root so config paths like
    # data/paper_trail/models/... do not resolve under training/paper_trail/.
    if kind == "dir":
        return root_candidate
    return root_candidate


def patch_trl_chunked_ce_for_partial_forward() -> None:
    """Make TRL chunked-CE work with PEFT `partial` forward + multi-GPU device_map."""
    import functools
    import inspect
    import types

    import trl.trainer.sft_trainer as sft_mod

    original_ce = sft_mod._chunked_cross_entropy_loss
    original_patch = sft_mod._patch_chunked_ce_lm_head

    def _device_aware_ce(hidden_states, lm_head_weight, chunk_size, **kwargs):
        # device_map=auto: last hidden often on cuda:1 while Trainer puts labels on cuda:0.
        device = hidden_states.device
        if lm_head_weight.device != device:
            hidden_states = hidden_states.to(lm_head_weight.device)
            device = hidden_states.device
        labels = kwargs.get("labels")
        if labels is not None and labels.device != device:
            kwargs["labels"] = labels.to(device)
        shift_labels = kwargs.get("shift_labels")
        if shift_labels is not None and shift_labels.device != device:
            kwargs["shift_labels"] = shift_labels.to(device)
        bias = kwargs.get("lm_head_bias")
        if bias is not None and bias.device != device:
            kwargs["lm_head_bias"] = bias.to(device)
        return original_ce(hidden_states, lm_head_weight, chunk_size, **kwargs)

    sft_mod._chunked_cross_entropy_loss = _device_aware_ce

    def _safe_patch(model, chunk_size, is_vlm: bool = False):
        fwd = model.forward
        if not hasattr(fwd, "__func__"):
            underlying = fwd
            while isinstance(underlying, functools.partial):
                underlying = underlying.func
            if hasattr(underlying, "__func__"):
                underlying = underlying.__func__

            def _forward_wrapper(self, *args, **kwargs):
                return fwd(*args, **kwargs)

            try:
                _forward_wrapper.__signature__ = inspect.signature(underlying)
            except (TypeError, ValueError):
                pass
            model.forward = types.MethodType(_forward_wrapper, model)
        return original_patch(model, chunk_size, is_vlm=is_vlm)

    sft_mod._patch_chunked_ce_lm_head = _safe_patch


def prime_cuda() -> None:
    """Ensure live CUDA contexts before importing transformers/trl.

    On this host (NVML driver mismatch + dual GPU), cold PeerToPeerAccess
    nvmlInit can assert. Skip if CUDA is already initialized. Touch every
    visible device so dual-GPU maps do not hit a cold peer-access init later.
    """
    if torch.cuda.is_initialized():
        print(
            f"cuda_prime skip-already-init count={torch.cuda.device_count()}",
            flush=True,
        )
        return
    n = torch.cuda.device_count()
    for i in range(max(n, 1)):
        _ = torch.empty(1, device=f"cuda:{i}" if n else "cuda")
    print(f"cuda_prime ok devices={n}", flush=True)


def build_dual_gpu_device_map(num_layers: int = 32) -> dict:
    """Put full-attn on 3090; a linear-attn block on 3060 for weight offload.

    Qwen3.5-4B full-attn every 4th layer dominates long-seq activation memory.
    Empirically ~3 linear layers on the 12GB card leave activation headroom for full-set ~20k.
    """
    # Only 3 linear layers on 3060 — leave activation headroom for full-set 20k.
    gpu1 = set(range(28, 31))
    device_map: dict = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        "model.norm": 0,
        "lm_head": 0,
    }
    for i in range(num_layers):
        device_map[f"model.layers.{i}"] = 1 if i in gpu1 else 0
    return device_map


def resolve_device_map(train_cfg: dict, quant: dict) -> dict | str | None:
    """Pick device_map for single- or dual-GPU (3090 24GB + 3060 12GB)."""
    raw = train_cfg.get("device_map")
    if raw in (None, "", "none"):
        raw = None
    multi = bool(train_cfg.get("multi_gpu", False))
    n = torch.cuda.device_count()
    if multi and n >= 2:
        max_memory = train_cfg.get("max_memory") or {
            0: "20GiB",
            1: "8GiB",
            "cpu": "80GiB",
        }
        norm: dict = {}
        for key, value in dict(max_memory).items():
            norm[int(key) if str(key).isdigit() else key] = value
        # Explicit map beats "auto" (auto parked almost all layers on the 3060).
        if raw in (None, "auto", "balanced"):
            device_map = build_dual_gpu_device_map(
                int(train_cfg.get("num_layers", 32))
            )
        else:
            device_map = raw
        print(
            f"multi_gpu device_map=explicit max_memory={norm} "
            f"gpu1_layers={[k for k,v in device_map.items() if v==1 and 'layers' in k]}",
            flush=True,
        )
        return {"device_map": device_map, "max_memory": norm}
    if quant.get("load_in_4bit"):
        return {"device_map": {"": 0}}
    if raw:
        return {"device_map": raw}
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs" / "paper_trail" / "train.toml",
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="optional cap on dataset rows (smoke / memory probes)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="resume training: omit path for latest checkpoint under output_dir, or pass a checkpoint path",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = repo_root()
    data_path = resolve_path(
        Path(args.data or cfg["data"]["train_jsonl"]), root, kind="file"
    )
    model_path = resolve_path(
        Path(args.model or cfg["model"]["path"]), root, kind="dir_existing"
    )
    output_dir = resolve_path(
        Path(args.output or cfg["train"]["output_dir"]), root, kind="dir"
    )

    if not data_path.is_file():
        raise SystemExit(f"missing dataset: {data_path}")
    if not model_path.is_dir():
        raise SystemExit(f"missing model: {model_path}")

    if not args.dry_run:
        prime_cuda()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    patch_trl_chunked_ce_for_partial_forward()

    dataset = load_dataset("json", data_files=str(data_path), split="train")

    def _force_no_thinking(example):
        # Qwen3.5 nothinking mode for SFT tokenization.
        kwargs = dict(example.get("chat_template_kwargs") or {})
        kwargs["enable_thinking"] = False
        example["chat_template_kwargs"] = kwargs
        # Defense in depth: drop any residual teacher thinking fields.
        messages = []
        for message in example.get("messages") or []:
            if not isinstance(message, dict):
                continue
            cleaned = {
                key: value
                for key, value in message.items()
                if key
                not in {
                    "reasoning_content",
                    "thinking",
                    "reason",
                    "reasoning",
                    "reasoning_details",
                }
            }
            messages.append(cleaned)
        example["messages"] = messages
        return example

    dataset = dataset.map(_force_no_thinking, desc="Force Qwen nothinking")
    if args.max_samples is not None and args.max_samples > 0:
        n = min(int(args.max_samples), len(dataset))
        dataset = dataset.select(range(n))
    print(f"samples={len(dataset)} model={model_path} out={output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    row0 = dataset[0]
    preview_kwargs = {
        "tools": row0.get("tools") or None,
        "tokenize": False,
        "add_generation_prompt": False,
    }
    try:
        preview = tokenizer.apply_chat_template(
            row0["messages"],
            enable_thinking=False,
            **preview_kwargs,
        )
    except TypeError:
        preview = tokenizer.apply_chat_template(row0["messages"], **preview_kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "template_preview.txt"
    preview_path.write_text(preview)
    print(f"template_preview_chars={len(preview)} -> {preview_path}")
    print(f"contains_<think>={('<think>' in preview)}")

    if args.dry_run:
        print("dry-run ok")
        return 0

    train_cfg = cfg["train"]
    max_steps = args.max_steps if args.max_steps is not None else train_cfg.get("max_steps")
    max_length = (
        int(args.max_length)
        if args.max_length is not None
        else int(train_cfg.get("max_length", 8192))
    )
    print(f"max_length={max_length} max_steps={max_steps}", flush=True)
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1.0)),
        max_steps=-1 if not max_steps else int(max_steps),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        save_steps=int(train_cfg.get("save_steps", 50)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        bf16=bool(train_cfg.get("bf16", True)),
        max_length=max_length,
        packing=False,
        assistant_only_loss=True,
        dataset_text_field=None,
        report_to=list(train_cfg.get("report_to") or []),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        warmup_steps=int(train_cfg.get("warmup_steps", 5)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        seed=int(train_cfg.get("seed", 7)),
    )

    dtype = torch.bfloat16 if train_cfg.get("bf16", True) else "auto"
    quant = cfg.get("quantization") or {}
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if quant.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if train_cfg.get("bf16", True)
            else torch.float16,
            bnb_4bit_use_double_quant=bool(quant.get("double_quant", True)),
            bnb_4bit_quant_type=str(quant.get("quant_type", "nf4")),
        )
    model_kwargs.update(resolve_device_map(train_cfg, quant))
    print(
        f"load model device_map={model_kwargs.get('device_map')} "
        f"quant_4bit={bool(quant.get('load_in_4bit'))} "
        f"visible_gpus={torch.cuda.device_count()}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    if getattr(model, "hf_device_map", None):
        print(f"hf_device_map={model.hf_device_map}", flush=True)
    if "device_map" not in model_kwargs:
        model = model.to("cuda")
    if hasattr(model, "config"):
        model.config.use_cache = False

    peft_cfg = None
    if cfg.get("lora", {}).get("enabled", True):
        from peft import LoraConfig

        lora = cfg["lora"]
        peft_kwargs = {
            "r": int(lora.get("r", 16)),
            "lora_alpha": int(lora.get("alpha", 32)),
            "lora_dropout": float(lora.get("dropout", 0.05)),
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": lora.get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        }
        if quant.get("load_in_4bit"):
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
        peft_cfg = LoraConfig(**peft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    resume_arg: bool | str | None = None
    if args.resume is not None:
        if args.resume in ("auto", "true", "1"):
            resume_arg = True
        else:
            resume_arg = str(resolve_path(Path(args.resume), root, kind="dir_existing"))
        print(f"resume_from_checkpoint={resume_arg}", flush=True)
    train_result = trainer.train(resume_from_checkpoint=resume_arg)
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(train_result.metrics, indent=2))
    print(f"saved adapter -> {output_dir / 'adapter'}")
    print(f"metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
