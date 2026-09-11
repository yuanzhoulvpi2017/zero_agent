"""Launch vLLM OpenAI server for PaperTrail SFT or base Qwen3.5-4B."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    import torch

    count = torch.cuda.device_count()
    if count < 1:
        raise SystemExit("no CUDA device visible")
    _ = torch.empty(1, device="cuda")
    print(f"warm cuda devices={count}", flush=True)

    # This host has an NVML driver/library mismatch; vLLM otherwise
    # treats CUDA as unavailable and stays on UnspecifiedPlatform.
    import vllm.platforms as vllm_platforms

    vllm_platforms.builtin_platform_plugins["cuda"] = (
        lambda: "vllm.platforms.cuda.CudaPlatform"
    )
    from vllm.platforms.cuda import CudaPlatform

    vllm_platforms.current_platform = CudaPlatform()
    print(f"vllm platform={type(vllm_platforms.current_platform).__name__}", flush=True)

    from vllm.entrypoints.cli.main import main as vllm_main

    root = repo_root()
    kind = os.environ.get("MODEL_KIND", "sft").strip().lower()
    if os.environ.get("MODEL"):
        model_path = Path(os.environ["MODEL"])
        default_name = os.environ.get("SERVED_MODEL_NAME") or model_path.name
    elif kind in {"base", "base-4b", "qwen"}:
        model_path = root / "model" / "Qwen" / "Qwen3.5-4B"
        default_name = "paper-trail-base"
    else:
        model_path = Path(
            os.environ.get(
                "MERGED",
                str(
                    root
                    / "data"
                    / "paper_trail"
                    / "models"
                    / "qwen35-4b-sft-qlora-16k"
                    / "merged"
                ),
            )
        )
        default_name = "paper-trail-sft"
    if not model_path.is_dir():
        hint = (
            "run: cd training/paper_trail && uv run python merge_lora.py"
            if kind not in {"base", "base-4b", "qwen"} and not os.environ.get("MODEL")
            else f"missing model directory: {model_path}"
        )
        raise SystemExit(f"missing checkpoint: {model_path}\n{hint}")
    print(f"serve kind={kind} model={model_path}", flush=True)

    sys.argv = [
        "vllm",
        "serve",
        str(model_path),
        "--served-model-name",
        os.environ.get("SERVED_MODEL_NAME", default_name),
        "--host",
        os.environ.get("HOST", "127.0.0.1"),
        "--port",
        os.environ.get("PORT", "8001"),
        "--device-ids",
        "0",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        os.environ.get("MAX_MODEL_LEN", "64000"),
        "--gpu-memory-utilization",
        os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"),
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        os.environ.get("REASONING_PARSER", "qwen3"),
        "--tool-call-parser",
        os.environ.get("TOOL_CALL_PARSER", "qwen3_coder"),
        "--language-model-only",
        "--default-chat-template-kwargs",
        '{"enable_thinking": false}',
    ]
    vllm_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
