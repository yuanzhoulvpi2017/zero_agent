#!/usr/bin/env bash
# Verify PaperTrail SFT outputs after a full train.
# Usage: ./verify_artifacts.sh [output_dir]
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
OUT="${1:-$ROOT/data/paper_trail/models/qwen35-4b-sft-qlora-16k}"
OUT="$(cd "$OUT" && pwd)"

fail() { echo "FAIL: $*" >&2; exit 1; }

[[ -f "$OUT/adapter/adapter_model.safetensors" ]] || fail "missing adapter weights"
[[ -f "$OUT/adapter/adapter_config.json" ]] || fail "missing adapter_config.json"
[[ -f "$OUT/train_metrics.json" ]] || fail "missing train_metrics.json"

python3 - <<PY
import json, sys
from pathlib import Path
out = Path("$OUT")
metrics = json.loads((out / "train_metrics.json").read_text())
epoch = float(metrics.get("epoch") or 0)
loss = metrics.get("train_loss")
weight = out / "adapter" / "adapter_model.safetensors"
size = weight.stat().st_size
print(f"adapter_bytes={size}")
print(f"train_loss={loss}")
print(f"epoch={epoch}")
print(f"keys={sorted(metrics)}")
if size < 1_000_000:
    raise SystemExit("adapter suspiciously small")
if epoch < 0.99:
    raise SystemExit(f"epoch incomplete: {epoch}")
print("OK")
PY

# Lightweight experiment manifest (AGENTS.md)
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
python3 - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
out = Path("$OUT")
metrics = json.loads((out / "train_metrics.json").read_text())
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "code_commit": "$COMMIT",
    "config": "configs/paper_trail/train_qlora_16k.toml",
    "base_model": "model/Qwen/Qwen3.5-4B",
    "dataset": "data/paper_trail/datasets/sft/train.jsonl",
    "max_length": 16384,
    "multi_gpu": True,
    "quantization": "nf4-qlora",
    "nothinking": True,
    "adapter_dir": str(out / "adapter"),
    "train_metrics": metrics,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"wrote {out / 'manifest.json'}")
PY
