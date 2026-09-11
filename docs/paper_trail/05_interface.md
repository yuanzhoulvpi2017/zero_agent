# PaperTrail 界面

本地网页是 PaperTrail 各阶段的统一入口。左侧是功能子页面，不是对话列表。采集阶段的对话、历史和轨迹已经可用；筛选标注、训练和评测先给出版面与输入输出约定，待对应阶段模块落地后再接真实操作。

## 启动

从仓库根目录：

```bash
# 已 export DEEPSEEK_API_KEY 时可跳过 .env
# 否则复制 configs/paper_trail/.env.example 为 .env 并填写
uv run python -m run_paper_trail

# 本地 SFT（需先启动 evaluation/paper_trail/serve_vllm.sh）
uv run python -m run_paper_trail --sft
```

访问 http://127.0.0.1:8000。界面使用 `interface/` 的虚拟环境，通过可编辑依赖安装 `agent/` 包。请使用单个 worker，以便进程内会话与缓存合并生效。

## 子页面

| 路径 | 阶段 | 状态 | 作用 |
| --- | --- | --- | --- |
| `/chat`、`/chat/{id}` | 采集 | 可用 | 与小埋多轮对话；每轮写入磁盘 |
| `/history`、`/history/{id}` | 采集 | 可用 | 回看用户可见正文，继续追问或删除 |
| `/traces`、`/traces/{id}` | 采集 | 可用 | 查看 thinking、工具调用和原始轨迹 |
| `/dataset` | 数据 | 规划中 | 过滤、标注、划分、导出 SFT |
| `/training` | 训练 | 规划中 | 提交训练、查看状态、导出权重 |
| `/evaluation` | 评测 | 规划中 | 同一测试集对比商业 / 基座 / SFT |

对话页右侧仍为工具调用。历史页只展示给用户看的回复；轨迹页读 `data/paper_trail/web/<session_id>/trajectory.json`。会话、缓存和模型都写在 `data/`，不提交 Git。新建对话不会删除旧记录。当前会话 ID 记在地址栏与浏览器本地存储中。

规划中的页面写明输入、输出和将提供的操作，不预建 `dataset/`、`training/`、`evaluation/` 空目录。实现时由对应阶段模块提供接口，界面只做展示与调用。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` 及上表子页面 | 同一套单页应用 |
| GET | `/api/sessions` | 对话摘要列表，按更新时间倒序 |
| GET | `/api/sessions/{id}` | 对话正文、工具卡片和渲染后的 Markdown |
| GET | `/api/sessions/{id}/trace` | 原始轨迹、配置与提交号 |
| POST | `/api/chat` | SSE：`session` / `message` / `done` / `error`；无 `session_id` 则新建 |
| GET | `/api/sessions/{id}/status` | 运行状态；磁盘中的历史会话也可查询 |
| GET | `/api/sessions/{id}/raw/{raw_ref}` | 该次工具调用的 Hugging Face 原始缓存 |
| DELETE | `/api/sessions/{id}` | 释放内存中的会话；`?purge=true` 同时删除磁盘记录 |

请求体：`{"session_id": "...", "message": "..."}`。消息 1–12000 字。进行中的会话返回 409。内存中闲置会话超过 50 个时会先卸下旧会话，磁盘记录保留；再次发送时从 `trajectory.json` 恢复记忆。

Markdown 在服务端渲染，原始 HTML 关闭。密钥不会写入轨迹。

## 检查

```bash
interface/.venv/bin/python interface/paper_trail/check_web.py
```

覆盖页面路由、空消息与缺失会话、流式事件与状态、失败后仍可查询会话、断线释放、原始缓存访问、Markdown 消毒，以及历史列表、轨迹读取、恢复后续聊和彻底删除。
