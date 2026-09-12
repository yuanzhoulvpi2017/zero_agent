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

打分也用 `deepseek-v4-flash`，维度：

| 维度 | 看什么 |
| --- | --- |
| tool_use | 该搜/读时是否调用工具，失败是否承认 |
| grounding | 论文身份和链接是否像真实检索，是否区分已读未读 |
| helpfulness | 是否对准用户当轮问题 |
| clarification | 含糊时是否追问，而不是一次堆论文 |
| persona_fit | 是否匹配对方知识水平和口气 |
| dialogue_quality | 是否承接上文、长度合适 |
| research_progress | 多轮后问题/对比/下一步是否更清楚 |
| identity | 被问身份时是否自称小埋；不编实验、不保证新颖 |

另统计工具次数、工具成功率、提到的 arXiv 数、耗时，以及同一人设上 SFT 对基座的配对胜率。

```bash
# 只看采样到的 16 条人设
python evaluation/paper_trail/run_paper_trail_eval.py --dry-sample

# 完整对比（先采 SFT，再切基座；单会话以免打满 3090）
# 需要 DEEPSEEK_API_KEY；3090 上的 :8001 会被脚本切换
python evaluation/paper_trail/run_paper_trail_eval.py

# 给蒸馏里的教师 flash 补打分（不重新对话）
python evaluation/paper_trail/run_paper_trail_eval.py \
  --run-dir data/paper_trail/eval/<run_id> --skip-collect --skip-vllm-swap --score-teacher

# 中断后续跑同一目录
python evaluation/paper_trail/run_paper_trail_eval.py --run-dir data/paper_trail/eval/<run_id>
```

产物：`manifest.json`、`personas.json`、各会话轨迹、`scores.json`、`report.md`。网页 `--sft` 在切换基座期间会暂时不可用。代码与本次数字见 [evaluation/paper_trail/README.md](../../evaluation/paper_trail/README.md)。

## 本次结果 · `compare-4b-20260911`

评委 `deepseek-v4-flash`。8 类 × 2 场 × 10 轮。4B 当场采集；教师从蒸馏轨迹抽样只评前 10 轮。

**教师 3.77，两个 4B 约 2.1–2.3。** SFT 工具行为靠近教师，依据和幻觉没有。

| 模型 | 总分 | 工具调用 | 读论文 | 幻觉论文标记 |
| --- | --- | --- | --- | --- |
| 教师 flash | **3.77 ± 0.40** | 18.8 | 81% | **2/16** |
| 未训 4B | 2.32 ± 0.22 | 3.5 | 25% | 15/16 |
| SFT 4B | 2.15 ± 0.35 | 9.3 | 75% | 16/16 |

完整表见 [evaluation/paper_trail/README.md](../../evaluation/paper_trail/README.md)。

## 检查清单

- [x] 合并 LoRA 为 vLLM 可加载权重
- [x] vLLM OpenAI 服务脚本（CUDA 预热 + 64k + qwen3 reasoning + qwen3_coder tools）
- [x] 网页 `--sft` 指向本地推理
- [x] 未训基座 vs SFT 的虚拟人批量对比（flash 打分）
- [x] 与教师 flash 同协议对比（复用蒸馏轨迹，只评前 10 轮）
