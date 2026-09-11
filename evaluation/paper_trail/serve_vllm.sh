#!/usr/bin/env bash
# Serve merged PaperTrail SFT on the 3090 (OpenAI-compatible).
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on this host.
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTORCH_CUDA_ALLOC_CONF || true
exec uv run python -u serve_vllm.py
