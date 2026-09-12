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

# 评委输入改过之后，用已有轨迹重打分（不重新对话）
python evaluation/paper_trail/run_paper_trail_eval.py \
  --run-dir data/paper_trail/eval/<run_id> --skip-collect --skip-vllm-swap --rejudge

# 中断后续跑
python evaluation/paper_trail/run_paper_trail_eval.py --run-dir data/paper_trail/eval/<run_id>
```

轨迹与原始分数在 `data/paper_trail/eval/`（不提交 Git）。3090 同时只够挂一个 64k 实例。

## 三个维度

总分只平均这三维。先前八维等权有问题：身份经常没被问却占 1/8；「对象适配」和「有用」重复；不调工具会在工具使用和依据上各扣一次。

| 字段 | 中文 | 看什么 | 不看什么 |
| --- | --- | --- | --- |
| grounding | 依据 | 该查时查了没有、失败是否承认、具体信息在不在工具证据里 | 话多话少、乱码、自称小埋 |
| helpfulness | 有用 | 对方已经说出口的事帮到没有；当轮接得住，整场有推进 | 论文号在不在工具里；清单太长 |
| dialogue | 对话 | 含糊先问清、长短可控、接上文、少乱码、不倾泻清单 | 有没有编论文；帮没帮上忙 |

工具次数、论文号依据率、被问身份时是否自称小埋，只做自动指标，不进总分。同一问题只进一维。

## 本次结果 · `compare-4b-20260911`

- 日期：2026-09-11（09-12 复评：补工具证据后，总分改为依据/有用/对话三维）
- 评委：`deepseek-v4-flash`
- 协议：8 类虚拟人各 2 场，每场 10 轮
- 4B：当场采集；教师：蒸馏 `constructed/` 中同协议抽样，**只评前 10 轮**，不重新对话（与 4B 不是同一段用户话）
- 基座：`model/Qwen/Qwen3.5-4B`；SFT：`qwen35-4b-sft-qlora-16k/merged`；教师：`deepseek-v4-flash`
- 原始报告：`data/paper_trail/eval/compare-4b-20260911/report.md`

**教师 4.69，SFT 2.06，基座 1.90。** 去掉身份和对象适配之后，SFT 总分略高于基座，主要赢在依据（更会查）；对话仍略差。教师三维都在 4 以上。

### 总分

| 模型 | n | 完成 | 总分均值 | 标准差 | 4B 同一人设胜场 |
| --- | --- | --- | --- | --- | --- |
| 教师 flash | 16 | 16 | **4.688** | 0.300 | — |
| SFT 4B | 16 | 16 | 2.062 | 0.475 | 6 胜 / 7 平（胜率 37.5%） |
| 未训 4B | 16 | 16 | 1.896 | 0.404 | 3 胜 |

### 分维度（1–5）

| 维度 | 教师 | 基座 | SFT | Δ(sft-base) |
| --- | --- | --- | --- | --- |
| 依据 | **4.750** | 1.562 | 2.062 | +0.500 |
| 有用 | **4.938** | 2.000 | 2.188 | +0.188 |
| 对话 | **4.375** | 2.125 | 1.938 | −0.187 |

### 自动指标（每场均值）

| 指标 | 教师 | 基座 | SFT |
| --- | --- | --- | --- |
| 工具调用次数 | **18.8** | 3.5 | 9.3 |
| 用过搜索 | **88%** | 69% | 81% |
| 用过读论文 | **81%** | 25% | 75% |
| 提到的 arXiv 数 | **6.9** | 1.2 | 5.7 |
| 论文号依据率 | **100%** | 88% | 74% |
| 助手字数 | 10585 | 12207 | **7225** |
| 耗时（秒，教师按 10/12 轮折算） | 168 | 87 | **71** |
| 评委幻觉论文标记 | **0/16** | 13/16 | 13/16 |
| 有工具外论文号的场次 | **0/16** | 2/16 | 10/16 |
| 被问身份时自称小埋 | 1/11 | 7/10 | 4/10 |

SFT 的搜/读已经靠近教师，论文号依据率 74%（教师 100%），依据分因此高于基座。基座更多无工具长篇讲解。SFT 仍会在工具外补作者、数字，倾泻清单 15/16，对话分略低。身份只记自动命中，不进总分。

### 按角色总分

| 角色 | 教师 | 基座 | SFT |
| --- | --- | --- | --- |
| 青年教师 | **4.667** | 2.333 | 1.500 |
| 跨方向转行者 | **4.667** | 2.000 | 2.833 |
| 在读硕士 | **4.500** | 1.500 | 2.167 |
| 独立研究者 | **4.500** | 2.000 | 2.000 |
| 算法工程师 | **4.833** | 1.833 | 2.333 |
| 博士生 | **4.500** | 1.833 | 2.000 |
| 组会质疑者 | **4.833** | 1.833 | 1.833 |
| 写综述的研究者 | **5.000** | 1.833 | 1.833 |
