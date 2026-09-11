# PaperTrail 数据处理 / 蒸馏采集

目标：用多样虚拟人与小埋对话，蒸馏 DeepSeek（当前配置为 `deepseek-v4-flash`）在 AgentScope 上的行为，为后续 Agent SFT 准备清晰对话链路。

## 虚拟人

代码：[`dataset/paper_trail_data/personas.py`](../../dataset/paper_trail_data/personas.py)

- 8 类角色：在读硕士、博士生、青年教师、算法工程师、写综述、跨方向转行、独立研究者、组会质疑者
- 采样时带主话题、岔开话题、口吻和 `jump_plan`（含身份/人物追问）
- **1 轮 = 用户一句 + 小埋一整轮回复**；单条对话最多 **20** 轮
- 虚拟人提示刻意压「需求说明书」口吻：目标只作内心动机，开口要短、含糊，让小埋追问后再慢慢补
- 每次采样写入 `prompt_version`（当前 `v6-human-shortchat-5`），便于按版本回溯构造数据

只看采样、不花额度：

```bash
uv run python -m run_paper_trail_collect --dry-sample --count 8 --seed 7
```

## 落盘内容（两边都记）

每次会话目录：

- 人工网页对话：`data/paper_trail/web/<session_id>/`
- **虚拟人构造**：`data/paper_trail/constructed/constructed__<persona_id>__<session_id>/`

一眼能看出是构造数据。目录内文件：

| 文件 | 内容 |
| --- | --- |
| `trajectory.json` | AgentScope **message** 侧：user / assistant / tool，含 thinking |
| `llm_calls.json` | **LLM** 侧：每次真正发给 DeepSeek 的 messages、tools，以及解析后的回复 |
| `dialogue_chain.json` | 按轮对齐的链路：每轮含 user、assistant 正文、该轮 agent_messages、该轮 llm_calls |
| `conversation.json` | 界面用摘要 |
| `persona.json` | 虚拟人设定（采集时） |
| `virtual_user_calls.json` | 虚拟人自己的 flash 调用（采集时） |
| `manifest.json` | 配置、用量、persona、moves |

## 采集

默认很省：1 条对话、最多 4 轮，模型读 `configs/paper_trail/agent.toml`（flash）。批量采集默认并行度 `2`，建议手动控制在 `2–3`。

```bash
# 先 dry-sample 确认角色
uv run python -m run_paper_trail_collect --dry-sample --count 1 --max-turns 3

# 真正调用模型（会消耗额度；终端有会话/轮次进度条）
uv run python -m run_paper_trail_collect --count 1 --max-turns 3 --seed 7

# 大批量示例
uv run python -m run_paper_trail_collect --count 10 --max-turns 20 --seed 7

# 更快一点：并行 3 条
uv run python -m run_paper_trail_collect --count 10 --max-turns 20 --seed 7 --concurrency 3

# 持续构造到额度不够自动停
uv run python -m run_paper_trail_collect --count 8 --max-turns 12 --seed 101 --concurrency 3 --loop
```

实现：[`agent/paper_trail/collect.py`](../../agent/paper_trail/collect.py)。采集时会打印总会话进度、总轮次进度，以及每条完成后的目录路径；需要完整 JSON 时加 `--json`。网页人工对话同样会写 `llm_calls.json` 与 `dialogue_chain.json`。

## 检查

```bash
interface/.venv/bin/python dataset/paper_trail_data/check_personas.py
interface/.venv/bin/python -m unittest paper_trail.checks.check_runtime
interface/.venv/bin/python dataset/paper_trail_data/quality.py
```

`quality.py` 会按 `prompt_version` 汇总开口长度、过长轮、复合句和背景泄露，结果写到 `data/paper_trail/collections/quality_report.json`。

## 转 SFT

把 `llm_calls.json` 转成 Qwen3.5 可用的对话样本（含 tools schema；**默认去掉 teacher thinking**）：

```bash
# 默认：去 thinking / reason / reasoning_content；按 64k token 预算智能压缩（先压旧 tool，再丢旧轮，不切最终 assistant）
cd training/paper_trail && uv run python ../../dataset/paper_trail_data/to_sft.py \
  --max-tokens 65536 \
  --model ../../model/Qwen/Qwen3.5-4B

interface/.venv/bin/python dataset/paper_trail_data/check_to_sft.py
```

输出：`data/paper_trail/datasets/sft/train.jsonl`。训练见 [03_training.md](03_training.md)。
