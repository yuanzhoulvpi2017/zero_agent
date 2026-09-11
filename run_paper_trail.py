"""Start PaperTrail from the repository root: uv run python -m run_paper_trail [--sft]."""

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
    argv = [item for item in sys.argv[1:] if item != "--sft"]
    sft = "--sft" in sys.argv[1:]
    environment = os.environ.copy()
    # The child uses interface/.venv, independent of the root tutorial environment.
    environment.pop("VIRTUAL_ENV", None)
    if sft:
        environment["PAPER_TRAIL_AGENT_CONFIG"] = str(
            root / "configs" / "paper_trail" / "agent_sft.toml"
        )
        print(
            "PaperTrail SFT：http://127.0.0.1:8000（需本机 vLLM :8001；Ctrl+C 停止）",
            flush=True,
        )
    else:
        print("PaperTrail：http://127.0.0.1:8000（Ctrl+C 停止）", flush=True)
    if argv:
        sys.exit(f"未知参数：{' '.join(argv)}")
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
