#!/usr/bin/env bash
# Launch PaperTrail SFT.
# - Default: both GPUs (3090+3060) for layer-split / QLoRA long context.
# - Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on this host
#   (triggers NVML PeerToPeerAccess assert).
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
unset PYTORCH_CUDA_ALLOC_CONF || true

ARGS_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")"

uv run python -u -c "
import json, runpy, sys, torch
n = torch.cuda.device_count()
for i in range(max(n, 1)):
    _ = torch.empty(1, device=f'cuda:{i}' if n else 'cuda')
print('warm devices', n, flush=True)
sys.argv = ['train.py'] + json.loads(r'''$ARGS_JSON''')
runpy.run_path('train.py', run_name='__main__')
"
