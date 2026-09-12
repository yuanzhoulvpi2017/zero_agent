# 面向算法的agent开发实战（python）

## 结构

| 名称 | 介绍 | B站视频 |
| --- | --- | --- |
| [code01_intro.md](https://github.com/yuanzhoulvpi2017/zero_agent/blob/main/code01_intro.md) | 介绍比较好用的，可以对标openclaw的agent开源框架（python） | [面向python、算法工程师的agent开发教程 s1-介绍](https://www.bilibili.com/video/BV12AwDzgEwe) |
| [code02_agent.md](https://github.com/yuanzhoulvpi2017/zero_agent/blob/main/code02_agent.md) | 调用大模型（LLM）和Agent的案例，包含MCP工具调用示例及Agent消息流程图、基于qwen-agent进行介绍 | [面向python、算法工程师的agent开发教程 s2-调用大模型和Agent](https://www.bilibili.com/video/BV1XADnBBEMy) |
| [code03_agentscope.md](https://github.com/yuanzhoulvpi2017/zero_agent/blob/main/code03_agentscope.mdd) | 对agentscope框架的介绍和使用 | [面向python、算法工程师的agent开发教程 s3-简单介绍一下agentscope的闪光点～](https://www.bilibili.com/video/BV1ksd9BcEx3) |

## 实战模块

基础教程讲完框架之后，这里用完整业务把「Agent → 轨迹 → 数据 → 训练 → 评测」跑通。每个业务在各阶段目录里用同一个英文名，产物进 `data/<业务名>/`，不提交 Git。新业务写法见 [docs/README.md](docs/README.md)。

代码按阶段放，不另建 `projects/<业务名>/`：

| 目录 | 用途 |
| --- | --- |
| [agent/](agent/README.md) | Agent、工具、轨迹采集 |
| [dataset/](dataset/README.md) | 清洗、过滤、SFT 格式转换 |
| [training/](training/README.md) | 训练与导出 |
| [evaluation/](evaluation/README.md) | 本地推理、批量评测 |
| [interface/](interface/README.md) | 网页与调用适配 |
| [configs/](configs/README.md) | 各阶段配置 |
| [docs/](docs/README.md) | 业务导航与分阶段教程 |

### 1. PaperTrail · 小埋论文探索助手

Agent 名叫「小埋」，由 B站 UP主「良睦路程序员」创建。目录和业务名仍用 `paper_trail`。

**要做什么：** 用户从研究兴趣出发，多轮检索、筛选、阅读 Hugging Face Papers 上的论文，梳理方法联系，形成有来源的研究问题和写作思路。同时把商业模型（当前教师是 `deepseek-v4-flash`）在这套工具上的行为蒸馏到本地 **Qwen3.5-4B**，再评测微调有没有把「会搜、会读、会聊」学下来。

**任务长什么样：** 「最近 Agent 长期记忆有点乱，先帮我摸几个分支。」开口可以很含糊；小埋应先检索、少堆论文、含糊时追问，并区分已读 / 未读。

#### 流程

```
真人网页 或 8 类虚拟人
        │
        ▼
  小埋 AgentScope 1.x
  教师：DeepSeek flash（thinking）
  工具：search / daily / metadata / read_paper / linked_resources
  只读 HF / arXiv，磁盘缓存
        │
        ▼
  轨迹双端落盘
  trajectory.json（message）
  llm_calls.json（真正发给模型的请求）
  dialogue_chain.json（一轮用户 + 一整轮回复）
        │
        ▼
  to_sft.py
  去 teacher thinking · Qwen3.5 tools 模板 · 只训 assistant
        │
        ▼
  Qwen3.5-4B 双卡 QLoRA（max_length=16384，1 epoch）
  merge LoRA → vLLM :8001
        │
        ▼
  同一套虚拟人协议对比 教师 flash / 未训 4B / SFT
  评委仍是 deepseek-v4-flash；教师列复用蒸馏轨迹打分
```

阶段可以反复跑，旧实验目录不覆盖。网页不必等训练完成，随时可对话、回看轨迹。

#### 各阶段在做什么

| 阶段 | 关键点 | 入口 |
| --- | --- | --- |
| Agent | 系统提示规定身份（小埋 / 良睦路程序员）、必须查工具才谈具体论文。工具返回精简字段，原文进缓存。 | [agent/paper_trail/](agent/paper_trail/)、[docs/paper_trail/01_agent.md](docs/paper_trail/01_agent.md) |
| 界面 | 本地流式聊天、历史、轨迹。`interface/` 只做展示和调用，不复制检索逻辑。 | [interface/run_paper_trail.py](interface/run_paper_trail.py)、[05_interface.md](docs/paper_trail/05_interface.md) |
| 采集 | 8 类虚拟人（硕/博/老师/工程师/综述/转行/独立/组会质疑），开口短、含糊，像微信。最多 20 轮。 | [interface/run_paper_trail_collect.py](interface/run_paper_trail_collect.py)、[02_dataset.md](docs/paper_trail/02_dataset.md) |
| 数据 | `llm_calls` → Qwen3.5 SFT jsonl；超长先压旧 tool 再丢旧轮，不切开最后一句 assistant。 | [dataset/paper_trail_data/to_sft.py](dataset/paper_trail_data/to_sft.py) |
| 训练 | TRL QLoRA，3090+3060 按层拆卡，默认 16k。不要设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。合并后再给 vLLM，不要 `--enable-lora`。 | [03_training.md](docs/paper_trail/03_training.md) |
| 评测 | 8 类 × 2 场 × 10 轮；虚拟用户始终 flash。4B 当场采集，教师复用蒸馏轨迹只评前 10 轮。打分维度：工具、来源、有用、澄清、人设、对话、研究推进、身份。 | [evaluation/paper_trail/README.md](evaluation/paper_trail/README.md) |

#### 怎么跑

需要 Python 3.12+ 与 uv。密钥用环境变量或 `configs/paper_trail/.env`，不要写进仓库。

```bash
# 1) 网页对话（教师 flash）
# export DEEPSEEK_API_KEY=...   或复制 configs/paper_trail/.env.example
python interface/run_paper_trail.py
# 浏览器 http://127.0.0.1:8000

# 2) 虚拟人蒸馏采集（先 dry-sample 看人设）
python interface/run_paper_trail_collect.py --dry-sample --count 1
python interface/run_paper_trail_collect.py --count 1 --max-turns 3

# 3) 转 SFT + 训练（见 03_training.md）后，合并 LoRA，起本地推理
cd evaluation/paper_trail && CUDA_VISIBLE_DEVICES=0 ./serve_vllm.sh
# 另开终端，网页走 SFT，不需要 DeepSeek 密钥
python interface/run_paper_trail.py --sft

# 4) 基座 4B vs SFT（会切换 :8001 上的模型）
python evaluation/paper_trail/run_paper_trail_eval.py --dry-sample
python evaluation/paper_trail/run_paper_trail_eval.py
# 教师列：复用蒸馏轨迹打分，不重新对话
python evaluation/paper_trail/run_paper_trail_eval.py \
  --run-dir data/paper_trail/eval/<run_id> --skip-collect --skip-vllm-swap --score-teacher
```

#### 当前进度与一次评测

- Agent、网页、虚拟人采集、SFT 转换、QLoRA 1 epoch、vLLM 部署、教师 / 基座 / SFT 同协议对比均已跑通。
- 教师对话走 `configs/paper_trail/agent.toml`；本地 SFT 走 `agent_sft.toml`（`http://127.0.0.1:8001/v1`，模型名 `paper-trail-sft`）。
- 一次对比（`compare-4b-20260911`，flash 评委，各 16 场）：教师 **3.77**，未训 4B **2.32**，SFT **2.15**。SFT 更会用工具（读论文 75% vs 基座 25%，教师 81%），但幻觉标记 16/16，教师只有 2/16。细节表见 [evaluation/paper_trail/README.md](evaluation/paper_trail/README.md)。

分阶段教程：[业务总览](docs/paper_trail/README.md) · [Agent](docs/paper_trail/01_agent.md) · [数据](docs/paper_trail/02_dataset.md) · [训练](docs/paper_trail/03_training.md) · [评测](docs/paper_trail/04_evaluation.md) · [界面](docs/paper_trail/05_interface.md)。

## 常见问题解答（FAQ）

### 当前 Agent 开发是否有必要使用现有框架？

**结论：建议优先使用成熟的开源框架，而非从零构建。**

尽管借助 AI 辅助编程工具（如 Codex、Claude Opus 等）可以快速实现一套自定义 Agent 框架，但这并不意味着此举是合理的工程选择。具体理由如下：

1. **社区验证与可靠性**：以 AgentScope、Qwen-Agent 为代表的主流框架，均已经过广泛的社区验证，存在公开的 Issue 追踪与修复记录。其核心开发者的工程能力普遍较强，框架的设计质量与稳定性有一定保障。
2. **AI 生成代码的局限性**：AI 虽然能够生成框架代码，但其输出并不保证完全正确，潜在的边界问题与隐性缺陷往往不会被主动暴露，存在一定的工程风险。
3. **重复造轮子的成本**：自研框架本质上仍需实现 LLM 响应解析、工具调用检测、循环调度、输出整理等基础能力，工程收益有限，属于典型的低价值重复劳动。
4. **Agent 开发的核心投入方向应聚焦于以下几点：**
   - 4.1 持续优化 Agent 的系统提示词（System Prompt）及工具定义（Tool Schema）；
   - 4.2 建立完善的测试体系，尤其是安全性测试，涵盖对抗性攻击、越权访问等场景；
   - 4.3 强化 Agent 的运行管理机制，包括状态管理、异步调用、并发控制与协程调度等；
   - 4.4 构建统一的 Agent 管理平台，对接多种框架，统一管控输入输出规范、并发限制与调用策略，实现框架无关的标准化治理。
