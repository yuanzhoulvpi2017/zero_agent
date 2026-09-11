"""Collect distillation dialogues from the repository root."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    uv = shutil.which("uv")
    if not uv:
        sys.exit("未找到 uv，请先安装 uv 并加入 PATH。")
    root = Path(__file__).resolve().parent
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
        cwd=root / "interface",
        env=environment,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
