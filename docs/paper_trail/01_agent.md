# PaperTrail Agent

小埋用 AgentScope 1.x 调用 DeepSeek（thinking 开启），通过只读 Hugging Face Papers 工具检索、阅读和比较论文。会话轨迹写入 `data/paper_trail/web/<session_id>/`，供界面回看和后续数据处理。该目录不提交 Git。

## 环境

在仓库根目录准备 Python 3.12+ 与 uv。密钥不要写入源码。

```bash
cp configs/paper_trail/.env.example configs/paper_trail/.env
# 填写 DEEPSEEK_API_KEY；可选 HF_TOKEN 以提高 Hugging Face 限额
uv sync --project agent --directory agent
```

配置见 [configs/paper_trail/agent.toml](../../configs/paper_trail/agent.toml)：当前教师模型为 `deepseek-v4-flash`。本地 SFT 推理改用 [agent_sft.toml](../../configs/paper_trail/agent_sft.toml)，通过环境变量 `PAPER_TRAIL_AGENT_CONFIG` 或 `uv run python -m run_paper_trail --sft` 切换。

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
