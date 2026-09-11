# PaperTrail 评测

本地 vLLM 推理，以及未训 Qwen3.5-4B 与 SFT 4B 接入小埋后的对比。业务说明见 [docs/paper_trail/04_evaluation.md](../../docs/paper_trail/04_evaluation.md)。

## 入口

| 文件 | 职责 |
| --- | --- |
| [serve_vllm.py](serve_vllm.py) / [serve_vllm.sh](serve_vllm.sh) | 在 3090 上提供 OpenAI 兼容接口。`MODEL_KIND=sft`（默认）或 `base` |
| [run_paper_trail_eval.py](run_paper_trail_eval.py) | 对比评测入口（走 interface 环境） |
| [compare_4b.py](compare_4b.py) | 虚拟人采集 + `deepseek-v4-flash` 打分 + 统计 |
| [scoring.py](scoring.py) | 自动指标与评委 JSON |

从仓库根目录：

```bash
# 只看 16 条人设
python evaluation/paper_trail/run_paper_trail_eval.py --dry-sample

# 完整对比（8 类 × 2 场 × 最多 10 轮；先 SFT 再基座；结束后恢复 SFT）
python evaluation/paper_trail/run_paper_trail_eval.py

# 中断后续跑
python evaluation/paper_trail/run_paper_trail_eval.py --run-dir data/paper_trail/eval/<run_id>
```

轨迹与原始分数在 `data/paper_trail/eval/`（不提交 Git）。3090 同时只够挂一个 64k 实例。

## 本次结果 · `compare-4b-20260911`

- 日期：2026-09-11
- 评委：`deepseek-v4-flash`
- 协议：8 类虚拟人各 2 场，每场 10 轮；虚拟用户始终是 flash
- 基座：`model/Qwen/Qwen3.5-4B`（`paper-trail-base`）
- SFT：`data/paper_trail/models/qwen35-4b-sft-qlora-16k/merged`（`paper-trail-sft`）
- 原始报告：`data/paper_trail/eval/compare-4b-20260911/report.md`

**SFT 更会用工具，但 flash 总分略低于基座。** 两者都约 2.1–2.3 / 5，主要扣在编造论文细节。

### 总分

| 模型 | n | 完成 | 总分均值 | 标准差 | 同一人设胜场 |
| --- | --- | --- | --- | --- | --- |
| 未训 4B | 16 | 16 | **2.320** | 0.221 | 10 胜 / 4 平 |
| SFT 4B | 16 | 16 | 2.148 | 0.349 | 2 胜（胜率 12.5%） |

### 分维度（1–5）

| 维度 | 基座 | SFT | Δ |
| --- | --- | --- | --- |
| 工具使用 | 1.750 | **1.938** | +0.188 |
| 澄清追问 | 2.312 | **2.500** | +0.188 |
| 来源依据 | **1.250** | 1.125 | −0.125 |
| 有用程度 | **2.125** | 2.062 | −0.063 |
| 人设适配 | **3.000** | 2.625 | −0.375 |
| 研究推进 | **2.062** | 1.750 | −0.312 |
| 对话质量 | **2.375** | 1.938 | −0.437 |
| 身份与边界 | **3.688** | 3.250 | −0.438 |

### 自动指标（每场均值）

| 指标 | 基座 | SFT |
| --- | --- | --- |
| 工具调用次数 | 3.5 | **9.3** |
| 用过搜索 | 69% | **81%** |
| 用过读论文 | 25% | **75%** |
| 提到的 arXiv 数 | 1.2 | **5.7** |
| 助手字数 | 12207 | **7225** |
| 耗时（秒） | 87 | **71** |
| 被问身份时自称小埋 | 7/10 | 4/10 |

评委给基座 15/16、SFT 16/16 场打了 `hallucinated_paper`。SFT 会搜会读、回复更短；仍常在工具结果之外编数字/作者，并有自相矛盾。基座则更多无工具长篇讲解。SFT 只在博士生（开源复现）和视觉转行者（多模态）两类上赢。

### 按角色总分

| 角色 | 基座 | SFT |
| --- | --- | --- |
| 青年教师 | **2.438** | 1.812 |
| 跨方向转行者 | 2.625 | **2.875** |
| 在读硕士 | 2.125 | 2.125 |
| 独立研究者 | **2.438** | 1.938 |
| 算法工程师 | **2.250** | 2.125 |
| 博士生 | 2.188 | **2.250** |
| 组会质疑者 | **2.250** | 2.125 |
| 写综述的研究者 | **2.250** | 1.938 |
