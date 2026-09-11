#!/usr/bin/env bash
# Serve PaperTrail SFT (default) or base 4B on the 3090 (OpenAI-compatible).
# MODEL_KIND=sft|base  MODEL=/path  SERVED_MODEL_NAME=...  PORT=8001
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on this host.
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTORCH_CUDA_ALLOC_CONF || true
exec uv run python -u serve_vllm.py
