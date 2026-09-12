"""Compare untrained Qwen3.5-4B vs SFT 4B on the same virtual-user protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
if str(DATASET) not in sys.path:
    sys.path.insert(0, str(DATASET))

from paper_trail.cache import ResponseCache  # noqa: E402
from paper_trail.collect import collect_one  # noqa: E402
from paper_trail.runtime import (  # noqa: E402
    constructed_root,
    data_root,
    load_settings,
    user_llm_settings,
)
from paper_trail_data import personas as personas_mod  # noqa: E402

from scoring import (  # noqa: E402
    DIMENSIONS,
    MODEL_ORDER,
    FlashJudge,
    automatic_metrics,
    render_report,
    summarize_scores,
)

EVAL_DIR = Path(__file__).resolve().parent
CONFIGS = {
    "base": ROOT / "configs" / "paper_trail" / "agent_base.toml",
    "sft": ROOT / "configs" / "paper_trail" / "agent_sft.toml",
}
SERVED_NAME = {
    "base": "paper-trail-base",
    "sft": "paper-trail-sft",
}
VLLM_URL = "http://127.0.0.1:8001/v1"
CONNECTION_MARKERS = (
    "connection error",
    "connection refused",
    "connect",
    "broken pipe",
    "server disconnected",
    "disconnected",
    "remote protocol",
    "read timeout",
    "timed out",
    "temporarily unavailable",
)


def is_connection_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in CONNECTION_MARKERS)


def eval_root() -> Path:
    return data_root() / "eval"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    for variable in ("DEEPSEEK_API_KEY", "LLM_API_KEY", "HF_TOKEN"):
        secret = os.getenv(variable)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def sample_eval_personas(seed: int, max_turns: int, replicas: int) -> list[dict]:
    people = []
    for index, category in enumerate(personas_mod.CATEGORIES):
        for replica in range(replicas):
            rng = random.Random(seed + index * 17 + replica)
            persona = personas_mod.sample_persona(rng, category, max_turns=max_turns)
            persona["eval_replica"] = replica
            people.append(persona)
    return people


def import_teacher_sessions(
    run_dir: Path,
    *,
    seed: int,
    replicas: int,
    scored_turns: int,
    min_turns: int = 10,
    max_source_turns: int = 12,
) -> list[dict]:
    """Reuse distillation constructed dialogues; do not call the teacher again."""
    pool: dict[str, list] = {key: [] for key in personas_mod.CATEGORIES}
    for path in constructed_root().iterdir():
        if not path.is_dir():
            continue
        persona_path = path / "persona.json"
        chain_path = path / "dialogue_chain.json"
        manifest_path = path / "manifest.json"
        if not (persona_path.is_file() and chain_path.is_file() and manifest_path.is_file()):
            continue
        try:
            persona = json.loads(persona_path.read_text())
            chain = json.loads(chain_path.read_text())
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        if manifest.get("status") != "completed":
            continue
        if (manifest.get("config") or {}).get("model") != "deepseek-v4-flash":
            continue
        category = persona.get("category")
        if category not in pool:
            continue
        turns = len(chain.get("turns") or [])
        if not min_turns <= turns <= max_source_turns:
            continue
        pool[category].append(
            {
                "directory": str(path),
                "persona_id": persona.get("id"),
                "category": category,
                "turns": turns,
                "model": "deepseek-v4-flash",
            }
        )
    rng = random.Random(seed)
    selected = []
    for category in personas_mod.CATEGORIES:
        items = list(pool[category])
        rng.shuffle(items)
        if len(items) < replicas:
            raise SystemExit(
                f"蒸馏数据中 {category} 可用场次不足 {replicas} "
                f"（需要 {min_turns}–{max_source_turns} 轮已完成会话）"
            )
        for replica, item in enumerate(items[:replicas]):
            selected.append(
                {
                    **item,
                    "storage_kind": "constructed",
                    "eval_replica": replica,
                    "scored_turns": scored_turns,
                    "dialogue_chain": str(Path(item["directory"]) / "dialogue_chain.json"),
                }
            )
    payload = {
        "model_tag": "teacher",
        "model": "deepseek-v4-flash",
        "source": "data/paper_trail/constructed",
        "scored_turns": scored_turns,
        "min_turns": min_turns,
        "max_source_turns": max_source_turns,
        "seed": seed,
        "note": "只评前 scored_turns 轮，不重新调用教师模型。",
        "count": len(selected),
        "results": selected,
    }
    teacher_dir = run_dir / "teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    write_json(teacher_dir / "collection.json", payload)
    tqdm.write(
        f"导入教师 {len(selected)} 条（每类 {replicas}，源轮次 {min_turns}–{max_source_turns}，"
        f"评分 {scored_turns} 轮）→ {teacher_dir / 'collection.json'}"
    )
    return selected


def find_completed(model_dir: Path, persona_id: str, max_turns: int) -> dict | None:
    if not model_dir.is_dir():
        return None
    for path in model_dir.iterdir():
        if not path.is_dir():
            continue
        persona_path = path / "persona.json"
        chain_path = path / "dialogue_chain.json"
        manifest_path = path / "manifest.json"
        if not (persona_path.is_file() and chain_path.is_file() and manifest_path.is_file()):
            continue
        try:
            persona = json.loads(persona_path.read_text())
            chain = json.loads(chain_path.read_text())
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        if persona.get("id") != persona_id:
            continue
        turns = chain.get("turns") or []
        if manifest.get("status") == "completed" and len(turns) >= max_turns:
            return {
                "session_id": manifest.get("session_id") or path.name.split("__")[-1],
                "directory": str(path),
                "storage_kind": "eval",
                "persona_id": persona_id,
                "turns": len(turns),
                "model": (manifest.get("config") or {}).get("model"),
                "dialogue_chain": str(chain_path),
                "resumed": True,
            }
    return None


def served_model_name(timeout: float = 5.0) -> str | None:
    try:
        response = httpx.get(f"{VLLM_URL}/models", timeout=timeout)
        response.raise_for_status()
        data = response.json().get("data") or []
        if not data:
            return None
        return data[0].get("id")
    except Exception:
        return None


def pids_on_port(port: int) -> list[int]:
    result = subprocess.run(
        ["ss", "-H", "-tlnp", f"sport = :{port}"],
        capture_output=True,
        text=True,
    )
    pids = []
    for match in __import__("re").findall(r"pid=(\d+)", result.stdout or ""):
        pids.append(int(match))
    return sorted(set(pids))


def gpu_used_mib() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return int(output.strip().splitlines()[0])
    except Exception:
        return None


def stop_vllm(port: int = 8001, wait_seconds: int = 60) -> None:
    pids = pids_on_port(port)
    if not pids:
        tqdm.write("vLLM 未在监听，无需停止。")
        return
    tqdm.write(f"停止 vLLM pid={pids}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if not pids_on_port(port):
            break
        time.sleep(1)
    leftover = pids_on_port(port)
    for pid in leftover:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        used = gpu_used_mib()
        if used is not None and used < 2500:
            break
        if not pids_on_port(port) and used is None:
            break
        time.sleep(1)
    tqdm.write(f"vLLM 已停止，GPU 占用约 {gpu_used_mib()} MiB")


def start_vllm(kind: str, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MODEL_KIND"] = kind
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    handle = log_path.open("ab")
    proc = subprocess.Popen(
        ["bash", str(EVAL_DIR / "serve_vllm.sh")],
        cwd=EVAL_DIR,
        env=environment,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    tqdm.write(f"启动 vLLM kind={kind} pid={proc.pid} log={log_path}")
    return proc


def wait_for_vllm(expected: str, timeout: int = 240) -> str:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = served_model_name(timeout=2.0)
        if last == expected:
            tqdm.write(f"vLLM 就绪：{last}")
            return last
        time.sleep(2)
    raise RuntimeError(f"等待 vLLM 超时，期望 {expected}，当前 {last}")


def ensure_vllm(kind: str, log_dir: Path, force: bool = False) -> None:
    expected = SERVED_NAME[kind]
    current = served_model_name()
    if current == expected and not force:
        tqdm.write(f"vLLM 已是 {current}，继续。")
        return
    if current and not force:
        tqdm.write(f"vLLM 当前是 {current}，切换到 {expected}。")
    else:
        tqdm.write(f"重启 vLLM → {expected}（当前 {current}）")
    stop_vllm()
    start_vllm(kind, log_dir / f"vllm-{kind}.log")
    wait_for_vllm(expected)


async def collect_model(
    *,
    run_dir: Path,
    model_tag: str,
    personas: list[dict],
    max_turns: int,
    concurrency: int,
    user_config: dict,
) -> list[dict]:
    config = load_settings(CONFIGS[model_tag])
    cache = ResponseCache(data_root() / "cache")
    model_dir = run_dir / model_tag
    model_dir.mkdir(parents=True, exist_ok=True)
    results = [None] * len(personas)
    pending = []
    for index, persona in enumerate(personas):
        existing = find_completed(model_dir, persona["id"], max_turns)
        if existing:
            results[index] = existing
        else:
            pending.append((index, persona))
    tqdm.write(
        f"采集 {model_tag}：跳过 {len(personas) - len(pending)} 条已完成，"
        f"待跑 {len(pending)} 条，并发 {concurrency}"
    )
    session_bar = tqdm(total=len(personas), desc=f"{model_tag} 会话", unit="条", dynamic_ncols=True)
    session_bar.update(len(personas) - len(pending))
    planned = sum(int(p["max_turns"]) for _, p in pending)
    turn_bar = tqdm(total=max(planned, 1), desc=f"{model_tag} 轮次", unit="轮", dynamic_ncols=True)
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    vllm_lock = asyncio.Lock()

    def note_turn(persona: dict, index: int, move: str) -> None:
        turn_bar.update(1)
        turn_bar.set_postfix_str(f"{persona['id'][:22]} · {index}/{persona['max_turns']} · {move}")

    async def run_one(index: int, persona: dict) -> None:
        async with semaphore:
            last_error = None
            for attempt in range(1, 4):
                try:
                    async with vllm_lock:
                        if served_model_name() != SERVED_NAME[model_tag]:
                            await asyncio.to_thread(ensure_vllm, model_tag, run_dir)
                    result = await collect_one(
                        persona,
                        config=config,
                        cache=cache,
                        on_turn_done=note_turn,
                        directory_root=model_dir,
                        user_config=user_config,
                        extra_manifest={
                            "storage_kind": "eval",
                            "evaluation": {
                                "task": "compare-4b",
                                "model_tag": model_tag,
                                "replica": persona.get("eval_replica"),
                            },
                        },
                    )
                    results[index] = result
                    async with lock:
                        session_bar.update(1)
                        tqdm.write(
                            f"✓ {model_tag} [{session_bar.n:.0f}/{len(personas)}] "
                            f"{result['persona_id']} → {result['directory']}"
                        )
                    return
                except Exception as exc:
                    last_error = exc
                    tqdm.write(
                        f"✗ {model_tag} {persona['id']} 第 {attempt} 次失败：{exc}"
                    )
                    if is_connection_error(exc) and attempt < 3:
                        async with vllm_lock:
                            await asyncio.to_thread(
                                ensure_vllm, model_tag, run_dir, True
                            )
                        await asyncio.sleep(2)
                        continue
                    break
            results[index] = {
                "persona_id": persona["id"],
                "error": str(last_error),
                "directory": None,
            }
            async with lock:
                session_bar.update(1)

    try:
        if pending:
            await asyncio.gather(*(run_one(index, persona) for index, persona in pending))
    finally:
        session_bar.close()
        turn_bar.close()
    payload = {
        "model_tag": model_tag,
        "model": config["model"],
        "count": len(results),
        "results": results,
    }
    write_json(model_dir / "collection.json", payload)
    return results


async def judge_run(run_dir: Path, user_config: dict, *, rejudge: bool = False) -> dict:
    scores_path = run_dir / "scores.json"
    judged_by_key = {}
    if rejudge and scores_path.is_file():
        backup = run_dir / "scores.prev.json"
        shutil.copy2(scores_path, backup)
        tqdm.write(f"已备份旧分数 → {backup}，按当前评委输入重打")
    elif scores_path.is_file():
        try:
            previous = json.loads(scores_path.read_text())
            for row in previous.get("sessions") or []:
                judged_by_key[(row["model_tag"], row["persona_id"])] = row
        except (OSError, ValueError):
            judged_by_key = {}
    rows = []
    pending = []
    for model_tag in MODEL_ORDER:
        collection_path = run_dir / model_tag / "collection.json"
        if not collection_path.is_file():
            continue
        collection = json.loads(collection_path.read_text())
        scored_turns = collection.get("scored_turns")
        for item in collection.get("results") or []:
            if not item or item.get("error") or not item.get("directory"):
                continue
            key = (model_tag, item["persona_id"])
            if key in judged_by_key:
                rows.append(judged_by_key[key])
            else:
                pending.append((model_tag, {**item, "scored_turns": item.get("scored_turns", scored_turns)}))
    tqdm.write(f"打分：已有 {len(rows)} 条，待评 {len(pending)} 条")
    judge = FlashJudge(user_config)
    try:
        for model_tag, item in tqdm(pending, desc="flash 打分", unit="条", dynamic_ncols=True):
            directory = Path(item["directory"])
            persona = json.loads((directory / "persona.json").read_text())
            scored_turns = item.get("scored_turns")
            auto = automatic_metrics(directory, max_turns=scored_turns)
            judged = await judge.judge_directory(directory, max_turns=scored_turns)
            row = {
                "model_tag": model_tag,
                "persona_id": persona["id"],
                "category": persona.get("category"),
                "category_label": persona.get("category_label"),
                "name": persona.get("name"),
                "topic": persona.get("topic"),
                "directory": str(directory),
                "scored_turns": scored_turns,
                "automatic": auto,
                **judged,
            }
            rows.append(row)
            write_json(
                run_dir / "scores.json",
                {
                    "judge_model": user_config["model"],
                    "dimensions": list(DIMENSIONS),
                    "sessions": rows,
                },
            )
    finally:
        await judge.close()
    summary = summarize_scores(rows)
    payload = {
        "judge_model": user_config["model"],
        "dimensions": DIMENSIONS,
        "sessions": rows,
        "summary": summary,
    }
    write_json(run_dir / "scores.json", payload)
    return payload


def print_stats(payload: dict) -> None:
    summary = payload.get("summary") or {}
    models = summary.get("models") or {}
    pairwise = summary.get("pairwise") or {}
    tqdm.write("")
    tqdm.write("======== 对比统计 ========")
    for tag in MODEL_ORDER:
        item = models.get(tag)
        if not item:
            continue
        tqdm.write(
            f"{tag}: n={item['n']} 完成={item['completed']} "
            f"总分={item['overall_mean']:.3f}±{item['overall_std']:.3f}"
        )
        for key, info in item["dimensions"].items():
            tqdm.write(f"  {key:20s} {info['mean']:.3f}")
    if pairwise.get("n"):
        tqdm.write(
            f"配对 {pairwise['n']}：SFT 胜 {pairwise['sft_wins']} / "
            f"基座胜 {pairwise['base_wins']} / 平 {pairwise['ties']} "
            f"（SFT 胜率 {pairwise.get('sft_win_rate')}）"
        )
    tqdm.write("==========================")


def build_manifest(run_dir: Path, args, personas: list[dict]) -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema": "paper_trail.eval.compare_4b.v1",
        "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "seed": args.seed,
        "max_turns": args.max_turns,
        "replicas": args.replicas,
        "concurrency": args.concurrency,
        "models": {
            "teacher": {
                "config": "configs/paper_trail/agent.toml",
                "checkpoint": None,
                "served_name": "deepseek-v4-flash",
                "source": "data/paper_trail/constructed（蒸馏已有轨迹，只评前 10 轮）",
            },
            "base": {
                "config": str(CONFIGS["base"].relative_to(ROOT)),
                "checkpoint": "model/Qwen/Qwen3.5-4B",
                "served_name": SERVED_NAME["base"],
            },
            "sft": {
                "config": str(CONFIGS["sft"].relative_to(ROOT)),
                "checkpoint": "data/paper_trail/models/qwen35-4b-sft-qlora-16k/merged",
                "served_name": SERVED_NAME["sft"],
            },
        },
        "judge_model": "deepseek-v4-flash",
        "persona_count": len(personas),
        "training": "qwen35-4b-sft-qlora-16k",
        "dataset": None,
        "evaluation": "compare-4b",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="对比基座 4B 与 SFT 4B 的小埋对话效果")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--replicas", type=int, default=2, help="每一类人设对话次数")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None, help="继续已有评测目录")
    parser.add_argument(
        "--models",
        default="sft,base",
        help="采集顺序，逗号分隔：sft,base（默认先用当前已启动的 SFT）",
    )
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument(
        "--score-teacher",
        action="store_true",
        help="从蒸馏 constructed/ 抽样教师 flash 场次并打分，不重新对话",
    )
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="忽略已有分数，按当前评委输入重打（旧 scores.json 备份为 scores.prev.json）",
    )
    parser.add_argument("--skip-vllm-swap", action="store_true", help="不启停 vLLM，假定端口已是对应模型")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条人设，调试用")
    parser.add_argument("--dry-sample", action="store_true")
    parser.add_argument("--keep-last-vllm", default="sft", help="全部采集结束后恢复的 vLLM 模型")
    return parser.parse_args(argv)


async def async_main(args) -> int:
    load_dotenv(ROOT / "configs/paper_trail/.env", override=False)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
    else:
        run_id = args.run_id or datetime.now().strftime("compare-4b-%Y%m%d-%H%M%S")
        run_dir = eval_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    personas_path = run_dir / "personas.json"
    if personas_path.is_file():
        personas = json.loads(personas_path.read_text())
        tqdm.write(f"沿用已有人设 {len(personas)} 条：{run_dir}")
    else:
        personas = sample_eval_personas(args.seed, args.max_turns, args.replicas)
        if args.limit is not None:
            personas = personas[: args.limit]
        write_json(personas_path, personas)
        tqdm.write(f"采样 {len(personas)} 条人设 → {personas_path}")
    if args.dry_sample:
        print(json.dumps(personas, ensure_ascii=False, indent=2))
        return 0

    manifest = build_manifest(run_dir, args, personas)
    write_json(run_dir / "manifest.json", manifest)
    user_config = user_llm_settings()
    if args.score_teacher:
        import_teacher_sessions(
            run_dir,
            seed=args.seed,
            replicas=args.replicas,
            scored_turns=args.max_turns,
        )
    model_tags = [item.strip() for item in args.models.split(",") if item.strip()]
    for tag in model_tags:
        if tag not in CONFIGS:
            raise SystemExit(f"未知模型标签：{tag}（可选 base,sft）")

    if not args.skip_collect:
        for tag in model_tags:
            if not args.skip_vllm_swap:
                ensure_vllm(tag, run_dir)
            else:
                current = served_model_name()
                tqdm.write(f"跳过 vLLM 切换，当前 {current}，采集 {tag}")
            await collect_model(
                run_dir=run_dir,
                model_tag=tag,
                personas=personas,
                max_turns=args.max_turns,
                concurrency=args.concurrency,
                user_config=user_config,
            )
        keep = args.keep_last_vllm
        if keep in SERVED_NAME and not args.skip_vllm_swap:
            try:
                ensure_vllm(keep, run_dir)
            except Exception as exc:
                tqdm.write(f"恢复 vLLM {keep} 失败：{exc}")

    if args.skip_judge:
        tqdm.write("已跳过打分。")
        return 0
    payload = await judge_run(run_dir, user_config, rejudge=args.rejudge)
    payload["run_id"] = run_dir.name
    payload["max_turns"] = args.max_turns
    report = render_report(payload)
    (run_dir / "report.md").write_text(report)
    print_stats(payload)
    tqdm.write(f"报告：{run_dir / 'report.md'}")
    tqdm.write(f"分数：{run_dir / 'scores.json'}")
    print(report)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        tqdm.write("收到中断。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
