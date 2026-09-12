# PaperTrail 评测

本地 vLLM 推理，以及教师 flash / 未训 Qwen3.5-4B / SFT 4B 接入小埋后的同协议对比。业务说明见 [docs/paper_trail/04_evaluation.md](../../docs/paper_trail/04_evaluation.md)。

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

# 给已有蒸馏轨迹里的教师 flash 打分（不重新对话，只评前 10 轮）
python evaluation/paper_trail/run_paper_trail_eval.py \
  --run-dir data/paper_trail/eval/<run_id> --skip-collect --skip-vllm-swap --score-teacher

# 中断后续跑
python evaluation/paper_trail/run_paper_trail_eval.py --run-dir data/paper_trail/eval/<run_id>
```

轨迹与原始分数在 `data/paper_trail/eval/`（不提交 Git）。3090 同时只够挂一个 64k 实例。

## 本次结果 · `compare-4b-20260911`

- 日期：2026-09-11（教师列于 09-12 补打分）
- 评委：`deepseek-v4-flash`
- 协议：8 类虚拟人各 2 场，每场 10 轮
- 4B：当场采集；教师：蒸馏 `constructed/` 中同协议抽样，**只评前 10 轮**，不重新对话（与 4B 不是同一段用户话）
- 基座：`model/Qwen/Qwen3.5-4B`；SFT：`qwen35-4b-sft-qlora-16k/merged`；教师：`deepseek-v4-flash`
- 原始报告：`data/paper_trail/eval/compare-4b-20260911/report.md`

**教师明显好于两个 4B（3.77 vs 2.32 / 2.15）。** SFT 比基座更会用工具，但依据和对话质量仍远低于教师；评委几乎只给两个 4B 打幻觉。

### 总分

| 模型 | n | 完成 | 总分均值 | 标准差 | 4B 同一人设胜场 |
| --- | --- | --- | --- | --- | --- |
| 教师 flash | 16 | 16 | **3.773** | 0.401 | — |
| 未训 4B | 16 | 16 | 2.320 | 0.221 | 10 胜 / 4 平 |
| SFT 4B | 16 | 16 | 2.148 | 0.349 | 2 胜（胜率 12.5%） |

### 分维度（1–5）

| 维度 | 教师 | 基座 | SFT | Δ(sft-base) |
| --- | --- | --- | --- | --- |
| 工具使用 | **3.562** | 1.750 | 1.938 | +0.188 |
| 来源依据 | **3.500** | 1.250 | 1.125 | −0.125 |
| 有用程度 | **4.125** | 2.125 | 2.062 | −0.063 |
| 澄清追问 | **3.688** | 2.312 | 2.500 | +0.188 |
| 人设适配 | **4.000** | 3.000 | 2.625 | −0.375 |
| 对话质量 | **4.125** | 2.375 | 1.938 | −0.437 |
| 研究推进 | **4.125** | 2.062 | 1.750 | −0.312 |
| 身份与边界 | 3.062 | **3.688** | 3.250 | −0.438 |

### 自动指标（每场均值）

| 指标 | 教师 | 基座 | SFT |
| --- | --- | --- | --- |
| 工具调用次数 | **18.8** | 3.5 | 9.3 |
| 用过搜索 | **88%** | 69% | 81% |
| 用过读论文 | **81%** | 25% | 75% |
| 提到的 arXiv 数 | **6.9** | 1.2 | 5.7 |
| 助手字数 | 10585 | 12207 | **7225** |
| 耗时（秒，教师按 10/12 轮折算） | 168 | 87 | **71** |
| `hallucinated_paper` | **2/16** | 15/16 | 16/16 |

SFT 的工具次数和读论文比例已经靠近教师，但来源依据（1.13 vs 教师 3.50）和幻觉率没有跟上来。教师也会 `dump_list`（10/16），真正拉开差距的是「工具里没有的细节不编」。

### 按角色总分

| 角色 | 教师 | 基座 | SFT |
| --- | --- | --- | --- |
| 青年教师 | **3.625** | 2.438 | 1.812 |
| 跨方向转行者 | **3.750** | 2.625 | 2.875 |
| 在读硕士 | **3.812** | 2.125 | 2.125 |
| 独立研究者 | **3.750** | 2.438 | 1.938 |
| 算法工程师 | **3.500** | 2.250 | 2.125 |
| 博士生 | **3.938** | 2.188 | 2.250 |
| 组会质疑者 | **4.250** | 2.250 | 2.125 |
| 写综述的研究者 | **3.562** | 2.250 | 1.938 |
