"""Start PaperTrail from the repository root: uv run python -m run_paper_trail."""

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
    print("PaperTrail：http://127.0.0.1:8000（Ctrl+C 停止）", flush=True)
    environment = os.environ.copy()
    # The child uses interface/.venv, independent of the root tutorial environment.
    environment.pop("VIRTUAL_ENV", None)
    try:
        result = subprocess.run(
            [
                uv,
                "run",
                "--locked",
                "python",
                "-m",
                "uvicorn",
                "app:app",
                "--app-dir",
                "paper_trail",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            cwd=root / "interface",
            env=environment,
        )
        return result.returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
