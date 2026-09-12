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

评委轨迹会带上工具返回的论文号和标题；回复里对得上的编号不算幻觉，2024–2026 年号不当成未来。另统计工具次数、论文号依据率、耗时，以及同一人设上 SFT 对基座的配对胜率。

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

**教师 4.69，SFT 2.06，基座 1.90。** SFT 依据高于基座（2.06 vs 1.56），对话略差。论文号依据率教师 100%、SFT 74%。

| 模型 | 总分 | 依据 | 有用 | 对话 | 幻觉论文标记 | 论文号依据率 |
| --- | --- | --- | --- | --- | --- | --- |
| 教师 flash | **4.69 ± 0.30** | **4.75** | **4.94** | **4.38** | **0/16** | **100%** |
| SFT 4B | 2.06 ± 0.48 | 2.06 | 2.19 | 1.94 | 13/16 | 74% |
| 未训 4B | 1.90 ± 0.40 | 1.56 | 2.00 | 2.13 | 13/16 | 88% |

完整表见 [evaluation/paper_trail/README.md](../../evaluation/paper_trail/README.md)。

## 检查清单

- [x] 合并 LoRA 为 vLLM 可加载权重
- [x] vLLM OpenAI 服务脚本（CUDA 预热 + 64k + qwen3 reasoning + qwen3_coder tools）
- [x] 网页 `--sft` 指向本地推理
- [x] 未训基座 vs SFT 的虚拟人批量对比（flash 打分）
- [x] 与教师 flash 同协议对比（复用蒸馏轨迹，只评前 10 轮）
- [x] 评委轨迹带工具论文号/标题，并按此复评已有场次
- [x] 打分收到三维（依据 / 有用 / 对话）；身份和工具次数只做自动指标
