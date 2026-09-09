# PaperTrail · 小埋论文探索助手

Agent 名称为「小埋」，由 B站 UP主「良睦路程序员」创建。项目与业务目录仍使用 PaperTrail / `paper_trail`。

基于 Hugging Face Papers，从研究兴趣出发，多轮检索、筛选和阅读论文，梳理方法联系，形成有来源依据的研究问题和写作思路。

## 当前范围

- [x] AgentScope 1.x + DeepSeek thinking + 原生 Python 工具。
- [x] 论文搜索、Daily Papers、元数据、分段阅读、关联模型/数据集/Spaces。
- [x] HF 成功响应的磁盘缓存与进程内同请求合并。
- [x] 本地多轮流式聊天网页、实时状态、会话轨迹，以及可回看、可继续的对话记录。
- [x] 虚拟人采样与蒸馏采集入口（message + LLM 双端落盘，≤20 轮）。
- [ ] 轨迹过滤、人工标注、SFT 数据生成。
- [ ] 模型训练与独立测试集评测。

代码：[Agent](../../agent/paper_trail/)、[界面](../../interface/paper_trail/)、[配置](../../configs/paper_trail/agent.toml)、[数据/蒸馏](../../dataset/paper_trail_data/)。

## 快速启动

以下命令从仓库根目录执行，需要 Python 3.12+ 与 uv。

```bash
# 已在 shell 中 export DEEPSEEK_API_KEY 时无需创建 .env。
# 也可复制 configs/paper_trail/.env.example 为 .env 并填写。
uv run python -m run_paper_trail
```

访问 http://127.0.0.1:8000。左侧是采集 / 数据 / 训练 / 评测子页面：当前可直接对话、查看历史对话和轨迹。示例问题：“我想了解最近 Agent 长期记忆方向，先帮我梳理几个分支。”每轮写入 `data/paper_trail/web/<session_id>/`，新建对话不会覆盖旧记录。`data/` 与模型文件不提交 Git。

蒸馏采集（虚拟人，默认 flash、少量轮次；写入 `data/paper_trail/constructed/constructed__<persona>__<session_id>/`）：

```bash
uv run python -m run_paper_trail_collect --dry-sample --count 1
uv run python -m run_paper_trail_collect --count 1 --max-turns 3
```

界面环境通过本地可编辑依赖安装 `agent/` 包，调用其公开的会话接口。仅开发后端时可运行 `uv sync --project agent`。根目录教程环境不受影响。

详细说明：[Agent 与工具](01_agent.md)、[数据与蒸馏](02_dataset.md)、[界面](05_interface.md)。
