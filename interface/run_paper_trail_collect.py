"""Collect PaperTrail distillation dialogues via the interface environment.

    python interface/run_paper_trail_collect.py --dry-sample --count 1
    python interface/run_paper_trail_collect.py --count 1 --max-turns 3
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

INTERFACE = Path(__file__).resolve().parent


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
            "-m",
            "paper_trail.collect",
            *sys.argv[1:],
        ],
        cwd=INTERFACE,
        env=environment,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
