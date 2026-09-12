# PaperTrail 评测与本地推理

用 vLLM 把训练好的 Qwen3.5-4B LoRA **合并后** 以 OpenAI 兼容接口提供给网页 Agent。SFT 为 nothinking（`enable_thinking=false`）。不要直接 `--enable-lora` 挂 adapter：Qwen3.5 的 GatedDeltaNet 对部分 LoRA 模块支持不完整。

## 产物

| 路径 | 说明 |
| --- | --- |
| `data/paper_trail/models/qwen35-4b-sft-qlora-16k/adapter/` | 训练写出的 LoRA |
| `data/paper_trail/models/qwen35-4b-sft-qlora-16k/merged/` | 合并后的 bf16 权重（vLLM 加载） |
| `configs/paper_trail/agent_sft.toml` | 网页走本地 vLLM |

## 启动推理服务

3090（`cuda:0`）即可；不要设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

```bash
# 若还没有 merged 目录
cd training/paper_trail
uv run python merge_lora.py

# 安装 vLLM（独立环境，Python 3.12）
cd ../../evaluation/paper_trail
uv sync
chmod +x serve_vllm.sh
CUDA_VISIBLE_DEVICES=0 ./serve_vllm.sh
```

默认 `http://127.0.0.1:8001/v1`，模型名 `paper-trail-sft`，`max_model_len=64000`，`--enable-auto-tool-choice --reasoning-parser qwen3 --tool-call-parser qwen3_coder --language-model-only`。

健康检查：

```bash
curl -s http://127.0.0.1:8001/v1/models
```

## 网页接 SFT

另开终端，仓库根目录：

```bash
python interface/run_paper_trail.py --sft
```

访问 http://127.0.0.1:8000。该模式读 `agent_sft.toml`，**不需要** DeepSeek 密钥。采集蒸馏仍用 `python interface/run_paper_trail.py` + `agent.toml`。

基座 4B 对比评测时用 `MODEL_KIND=base ./serve_vllm.sh`，配置见 [agent_base.toml](../../configs/paper_trail/agent_base.toml)。3090 同时只够挂一个 64k 实例，脚本会按 `sft → base` 切换。

## 基座 4B vs SFT vs 教师 flash

协议：8 类虚拟人各 2 场，每场最多 10 轮。虚拟用户始终是 `deepseek-v4-flash`。小埋分别接未训练 Qwen3.5-4B 与合并后的 SFT（当场采集）。教师列**复用蒸馏已有轨迹**（`data/paper_trail/constructed/`，同 8 类 × 2 场、只评前 10 轮），不重新调用 flash 当助手。工具、系统提示对齐；4B 与教师不是同一段用户话，只对齐协议与评委。

打分也用 `deepseek-v4-flash`。总分只平均三维；八维等权已弃用（身份常没被问、对象适配和有用重复、不调工具会重复扣分）。

| 字段 | 中文 | 看什么 | 不看什么 |
| --- | --- | --- | --- |
| grounding | 依据 | 该查时是否查了、失败是否承认、论文号/作者/数字是否在工具证据里 | 话多话少、乱码、自称小埋 |
| helpfulness | 有用 | 对方已经说出口的事帮到没有；当轮接得住，整场有推进 | 论文号在不在工具里；清单太长 |
| dialogue | 对话 | 含糊先问清、长短可控、接上文、少乱码、不倾泻清单 | 有没有编论文；帮没帮上忙 |

评委轨迹会带上工具返回的论文号和标题；回复里对得上的编号不算幻觉，2024–2026 年号不当成未来。诊断指标分开报告：

- 工具行为：调用次数、使用搜索/阅读的场次、全部工具结果的微平均成功率；只描述行为，不表示越多越好。
- 明确 arXiv ID 核验：工具内 ID / 回复全部明确 ID，按全部 ID 微平均；不覆盖无编号标题、作者、数字和仓库。
- 输出特征：每轮助手字符数；不是中文分词后的字数。
- 评委告警：幻觉论文、倾泻清单、该查未查；是模型判断，不是自动指标。
- 条件场景：只在虚拟用户明确问身份时检查是否自称小埋，不进总分。

教师轨迹原本有 10–12 轮，按 10/12 折算耗时与 4B 实测耗时不可直接比较，因此不再放进主结果。

读表时必须注意：

1. 只有依据、有用、对话进入总分，其他指标只用于解释原因。
2. 每个模型只有 16 场，结果用于定位问题，不是稳定排行榜。
3. 教师与 4B 不是相同用户逐句配对；只对齐角色类别、前 10 轮和评委。配对胜负只用于基座 vs SFT。
4. 工具调用多不代表更好；工具成功只表示接口返回 `ok=true`。
5. 明确 ID 支持率只覆盖回复里写出的 arXiv ID，不是完整事实正确率。
6. 幻觉、倾泻和该查未查是评委告警，不是自动事实。
7. 教师和评委都是 `deepseek-v4-flash`，可能存在同模型风格偏好；教师分数不是独立评委下的绝对值。

```bash
# 只看采样到的 16 条人设
python evaluation/paper_trail/run_paper_trail_eval.py --dry-sample

# 完整对比（先采 SFT，再切基座；单会话以免打满 3090）
# 需要 DEEPSEEK_API_KEY；3090 上的 :8001 会被脚本切换
python evaluation/paper_trail/run_paper_trail_eval.py

# 给蒸馏里的教师 flash 补打分（不重新对话）
python evaluation/paper_trail/run_paper_trail_eval.py \
  --run-dir data/paper_trail/eval/<run_id> --skip-collect --skip-vllm-swap --score-teacher

# 评委输入改过之后重打分（旧 scores.json 备份为 scores.prev.json）
python evaluation/paper_trail/run_paper_trail_eval.py \
  --run-dir data/paper_trail/eval/<run_id> --skip-collect --skip-vllm-swap --rejudge

# 中断后续跑同一目录
python evaluation/paper_trail/run_paper_trail_eval.py --run-dir data/paper_trail/eval/<run_id>
```

产物：`manifest.json`、`personas.json`、各会话轨迹、`scores.json`、`report.md`。网页 `--sft` 在切换基座期间会暂时不可用。代码与本次数字见 [evaluation/paper_trail/README.md](../../evaluation/paper_trail/README.md)。

## 本次结果 · `compare-4b-20260911`

评委 `deepseek-v4-flash`。8 类 × 2 场 × 10 轮。4B 当场采集；教师从蒸馏轨迹抽样只评前 10 轮。总分三维：依据 / 有用 / 对话。

**教师 4.69，SFT 2.06，基座 1.90。具体解释：**

- 教师 111 个明确 ID 全在工具结果中，工具结果成功率 99%；评委给 12/16 场依据 5 分、15/16 场有用 5 分。它常在读取失败时承认限制，并按用户要求收短。不过教师不是和 4B 逐句配对，且由同一个 flash 评审，4.69 可能受同模型风格偏好影响。
- SFT 依据高于基座 0.50，主要因为每场工具调用 9.3 vs 3.5、读论文场次 12/16 vs 4/16、该查未查告警 2/16 vs 6/16。它是更常查，不是事实更准：明确 ID 支持率 SFT 71.4%、基座 78.9%，两者都有 13/16 场幻觉告警。
- 有用分差 +0.19、对话分差 −0.19，都只相当于 16 场中总共相差 3 个单点评分，不应当作稳定胜负。SFT 回复更短，但倾泻清单告警达到 15/16，并有乱码和绕圈。
- SFT 对基座 6 胜、3 负、7 平，只能说本批样本轻微领先。训练明确学到了工具行为，没有学稳可靠引用和对话收敛。

| 模型 | 总分 | 依据 | 有用 | 对话 | 幻觉论文告警 | 明确 ID 支持率 |
| --- | --- | --- | --- | --- | --- | --- |
| 教师 flash | **4.69 ± 0.30** | **4.75** | **4.94** | **4.38** | **0/16** | **100%** |
| SFT 4B | 2.06 ± 0.48 | 2.06 | 2.19 | 1.94 | 13/16 | 71.4% |
| 未训 4B | 1.90 ± 0.40 | 1.56 | 2.00 | 2.13 | 13/16 | 78.9% |

完整表见 [evaluation/paper_trail/README.md](../../evaluation/paper_trail/README.md)。

## 检查清单

- [x] 合并 LoRA 为 vLLM 可加载权重
- [x] vLLM OpenAI 服务脚本（CUDA 预热 + 64k + qwen3 reasoning + qwen3_coder tools）
- [x] 网页 `--sft` 指向本地推理
- [x] 未训基座 vs SFT 的虚拟人批量对比（flash 打分）
- [x] 与教师 flash 同协议对比（复用蒸馏轨迹，只评前 10 轮）
- [x] 评委轨迹带工具论文号/标题，并按此复评已有场次
- [x] 打分收到三维（依据 / 有用 / 对话）；身份和工具次数只做自动指标
