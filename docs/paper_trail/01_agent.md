# PaperTrail Agent

小埋用 AgentScope 1.x 调用 DeepSeek（thinking 开启），通过只读 Hugging Face Papers 工具检索、阅读和比较论文。会话轨迹写入 `data/paper_trail/web/<session_id>/`，供界面回看和后续数据处理。该目录不提交 Git。

## 环境

在仓库根目录准备 Python 3.12+ 与 uv。密钥不要写入源码。

```bash
cp configs/paper_trail/.env.example configs/paper_trail/.env
# 填写 DEEPSEEK_API_KEY；可选 HF_TOKEN 以提高 Hugging Face 限额
uv sync --project agent --directory agent
```

配置见 [configs/paper_trail/agent.toml](../../configs/paper_trail/agent.toml)：当前教师模型为 `deepseek-v4-flash`。本地 SFT 推理改用 [agent_sft.toml](../../configs/paper_trail/agent_sft.toml)，通过环境变量 `PAPER_TRAIL_AGENT_CONFIG` 或 `python interface/run_paper_trail.py --sft` 切换。

## 代码

| 文件 | 职责 |
| --- | --- |
| [runtime.py](../../agent/paper_trail/runtime.py) | 会话、流式事件、DeepSeek thinking 对齐、轨迹与对话记录 |
| [tools.py](../../agent/paper_trail/tools.py) | 搜索、Daily Papers、元数据、分段阅读、关联资源 |
| [compact.py](../../agent/paper_trail/compact.py) | 工具返回给模型的精简字段；原始响应当场缓存 |
| [cache.py](../../agent/paper_trail/cache.py) | 成功 GET 的磁盘缓存与进程内同请求合并 |

工具只接受 Hugging Face / arXiv 的 HTTPS 链接或现代 arXiv ID。Daily Papers 是社区精选，不能当成全部最新论文。`read_paper` 按字符窗口返回 Markdown，长文用 `next_offset` 续读；返回内容可能只是介绍页。

## 产物

每个会话目录包含：

- `manifest.json`：会话 ID、标题、提交、配置、状态、用量；不含密钥
- `trajectory.json`：AgentScope 消息（含 thinking、工具调用与返回）
- `conversation.json`：界面用摘要（用户/助手正文、工具卡片、用量；不含 thinking）

成功的 Hugging Face 响应缓存在 `data/paper_trail/cache/`。`data/` 与模型权重不提交 Git。

每轮对话还会写入：

- `trajectory.json`：AgentScope message 轨迹
- `llm_calls.json`：发给 DeepSeek 的请求与解析后的回复
- `dialogue_chain.json`：按「用户一句 + 小埋一轮」对齐的 SFT 友好链路

蒸馏采集见 [02_dataset.md](02_dataset.md)。

## 检查

```bash
interface/.venv/bin/python -m unittest paper_trail.checks.check_compact paper_trail.checks.check_runtime
```

覆盖缓存合并与 TTL、工具截断与 ID 校验、thinking 在多轮 assistant 消息中的对齐、流式工具循环，以及对话标题、conversation.json 与磁盘恢复。

界面启动与 API 见 [05_interface.md](05_interface.md)。

## 教学示例：逐次保存模型调用

- [x] [code001.py](../../agent/paper_trail/simple/code001.py) 接入 [RecordingModel](../../agent/paper_trail/simple/recording_model.py)，保留原有聊天与工具流程。

在仓库根目录运行（密钥配置同上）：

```bash
uv run --project agent agent/paper_trail/simple/code001.py
```

终端会显示记录目录，默认是 `data/paper_trail/simple/<随机会话ID>/`。`RecordingModel(directory=Path(...), **模型参数)` 可指定目录；相对路径按仓库根目录解析。每次模型调用保存 `turn_001_call_001_<唯一ID>.json`，中文直接显示并缩进。一句用户输入可能触发多次模型调用，它们的 `turn_index` 相同、`call_index` 递增。

`manifest_<唯一ID>.json` 记录代码提交与实际配置，尚未执行的数据处理、训练、评测留空。调用 JSON 包含模型标识、时间、请求消息与工具定义、回复正文、thinking、工具调用、token 用量、耗时和执行状态。工具返回出现在下一次模型请求的 `request.messages` 中。流式输出原样传递给 Agent；文件先记录 running，再在结束时保存最终累计结果，失败或中断也保留已获得的内容。此记录是模型调用视角，尚未再次发送给模型的工具结果不会包含在其中。真实密钥会脱敏，产物不提交 Git。

### VS Code 调试教学示例

- [x] [.vscode/launch.json](../../.vscode/launch.json) 提供 `PaperTrail: 调试 code001`。

用 VS Code 打开仓库根目录，安装 Python 与 Python Debugger 扩展。在“运行和调试”中选择该启动项，设置断点后按 F5；对话输入在集成终端进行。配置使用根目录 `.venv/bin/python`（对应根目录的 `uv run` 环境），工作目录固定为仓库根目录，并通过 `PYTHONPATH` 加入 `agent/`。环境变量可由 `configs/paper_trail/.env` 提供。`justMyCode: false` 允许进入 AgentScope 等依赖源码。
