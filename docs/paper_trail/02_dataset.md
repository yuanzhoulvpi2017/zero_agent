# PaperTrail 数据处理 / 蒸馏采集

目标：用多样虚拟人与小埋对话，蒸馏 DeepSeek（当前配置为 `deepseek-v4-flash`）在 AgentScope 上的行为，为后续 Agent SFT 准备清晰对话链路。

## 虚拟人

代码：[`dataset/paper_trail_data/personas.py`](../../dataset/paper_trail_data/personas.py)

- 8 类角色：在读硕士、博士生、青年教师、算法工程师、写综述、跨方向转行、独立研究者、组会质疑者
- 采样时带主话题、岔开话题、口吻和 `jump_plan`（含身份/人物追问）
- **1 轮 = 用户一句 + 小埋一整轮回复**；单条对话最多 **20** 轮
- 虚拟人提示刻意压「需求说明书」口吻：目标只作内心动机，开口要短、含糊，让小埋追问后再慢慢补
- 虚拟人按会话保存双方完整对话，每轮以 JSON 聊天记录传入此前自己的发言和小埋的全部回复，不截断；明确标注 user 为虚拟人、assistant 为小埋。轮次引导和改写请求不作为真实聊天历史，历史只记录最终发出的虚拟人短句；短句要求仅约束虚拟人的输出。
- 每次采样写入 `prompt_version`（当前 `v6-human-shortchat-5`），便于按版本回溯构造数据

只看采样、不花额度：

```bash
python interface/run_paper_trail_collect.py --dry-sample --count 8 --seed 7
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
python interface/run_paper_trail_collect.py --dry-sample --count 1 --max-turns 3

# 真正调用模型（会消耗额度；终端有会话/轮次进度条）
python interface/run_paper_trail_collect.py --count 1 --max-turns 3 --seed 7

# 大批量示例
python interface/run_paper_trail_collect.py --count 10 --max-turns 20 --seed 7

# 更快一点：并行 3 条
python interface/run_paper_trail_collect.py --count 10 --max-turns 20 --seed 7 --concurrency 3

# 持续构造到额度不够自动停
python interface/run_paper_trail_collect.py --count 8 --max-turns 12 --seed 101 --concurrency 3 --loop
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

### simple 日志转换教学示例

- [x] [code002_to_sft.py](../../agent/paper_trail/simple/code002_to_sft.py)：一个入口完成原始 model logs → SFT 清洗 → **全部会话、全部调用**的 loss mask 检查；原 code003 已合并移除。

从仓库根目录运行，复用训练环境中的 Transformers / PyTorch，仅加载本地 tokenizer，不加载模型或训练：

```bash
training/paper_trail/.venv/bin/python agent/paper_trail/simple/code002_to_sft.py
# 指定一个会话；输出目录必须尚不存在
training/paper_trail/.venv/bin/python agent/paper_trail/simple/code002_to_sft.py \
  --input data/paper_trail/simple/<session_id> \
  --output-dir data/paper_trail/mask_demo/my_demo
```

相对路径以仓库根目录解析，与执行目录无关。默认输入为 `data/paper_trail/simple/`，递归读取所有 `turn_*_call_*.json`，排除虚拟人调用、conversation 和 manifest。仅使用 `status=completed` 且请求和回复非空的调用；纯工具调用也有效。未完成或空调用计入跳过统计；损坏日志、缺失 schema 或 token 前缀不匹配均报出来源路径，不静默漏掉。所有样本通过转换及前缀检查后才写出新批次。

清洗规则：去掉目标和历史 assistant 的 reasoning；**任何有 tool_calls 的 assistant 都清空 content，只保留结构化调用和参数**。不带 tool_calls 的 assistant 保留回复正文。工具返回、工具 schema 和原始日志不改。每份日志转为一条样本，messages 为清洗后的历史加本次输出，tools 保留完整定义。

默认输出 `data/paper_trail/mask_demo/<时间_UUID>/`：

| 文件 / 目录 | 内容 |
| --- | --- |
| `README.md` / `index.json` | 全部会话和调用的索引，逐条链接检查结果 |
| `samples.jsonl` | 全量清洗后 SFT messages、tools、来源元信息 |
| `encoded.jsonl` | 与 samples 逐行对应的 input_ids、attention_mask、loss_mask、labels |
| `all_mask_segments.txt` | 所有样本的 mask 区段汇总 |
| `sessions/<session_id>/<日志名>/` | 每次调用的 sample.json、batch.json、tokens.tsv、rendered.txt、prompt.txt、mask_segments.txt、supervised.txt |
| `original_template.jinja` | 未经修改的模型原始模板 |
| `manifest.json` | 来源哈希、实际配置、代码版本、会话数、SFT 数、mask 数和跳过统计 |

`data/paper_trail/mask_demo/README.md` 指向当前全量批次。旧目录保留为历史记录，包含不同旧模板和单条示例，不再作为当前检查依据。原始日志保持不变，不覆盖旧批次。

每次调用只监督本次末尾输出；工具调用样本只监督调用本身，回复样本监督正文。训练接入时必须保留 `encoded.jsonl` 的 labels；直接把 `samples.jsonl` 交给监督所有 assistant 的 Trainer，会改变这里展示的 mask。token 前缀算法及空 think 边界见 [03_training.md](03_training.md#loss-mask-教学示例仅分词)。

产物尚未划分训练 / 测试集；训练前按会话或任务留出测试集，并检查目标模型长度。当前示例保留完整序列，不截断。

### 批量采集日志转换

把 `llm_calls.json` 转成 Qwen3.5 可用的对话样本（含 tools schema；**默认去掉 teacher thinking**）：

```bash
# 默认：去 thinking / reason / reasoning_content；按 64k token 预算智能压缩（先压旧 tool，再丢旧轮，不切最终 assistant）
cd training/paper_trail && uv run python ../../dataset/paper_trail_data/to_sft.py \
  --max-tokens 65536 \
  --model ../../model/Qwen/Qwen3.5-4B

interface/.venv/bin/python dataset/paper_trail_data/check_to_sft.py
```

输出：`data/paper_trail/datasets/sft/train.jsonl`。训练见 [03_training.md](03_training.md)。
