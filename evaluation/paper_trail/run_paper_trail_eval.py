"""Run PaperTrail 4B base vs SFT comparison via the interface environment.

    python evaluation/paper_trail/run_paper_trail_eval.py --dry-sample
    python evaluation/paper_trail/run_paper_trail_eval.py
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INTERFACE = ROOT / "interface"
COMPARE = HERE / "compare_4b.py"


def main() -> int:
    uv = shutil.which("uv")
    if not uv:
        sys.exit("未找到 uv，请先安装 uv 并加入 PATH。")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    result = subprocess.run(
        [
            uv,
            "run",
            "--locked",
            "python",
            str(COMPARE),
            *sys.argv[1:],
        ],
        cwd=INTERFACE,
        env=environment,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
