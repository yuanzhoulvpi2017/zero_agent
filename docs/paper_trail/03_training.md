# PaperTrail 训练

目标：用构造轨迹蒸馏出的 SFT 数据，在本地 Qwen3.5-4B 上做 tool-calling Agent 微调（TRL + QLoRA）。

## 数据

```bash
# 推荐：用训练环境的 transformers 做预算拟合；默认去掉 teacher thinking
cd training/paper_trail
uv run python ../../dataset/paper_trail_data/to_sft.py \
  --max-tokens 65536 \
  --model ../../model/Qwen/Qwen3.5-4B \
  --preview-template

# 自检
cd ../.. && interface/.venv/bin/python dataset/paper_trail_data/check_to_sft.py
```

每条样本 = 一次 teacher `llm_calls` 的**完整前后文**（request.messages + 本步 assistant）+ `tools`。

- Mask：TRL `assistant_only_loss=True`（只训 assistant / `<tool_call>`）
- **默认去掉** DeepSeek-Flash 的 `thinking` / `reasoning_content` / `reason`
- 训练侧强制 Qwen3.5 **nothinking**（`enable_thinking=False`）
- 超长时：先压缩最旧 tool，再丢旧轮；**不切开**最终 assistant
- 去 thinking 后长度约 p50≈26k / p90≈33k / max≈42k（无 64k 样本）

## 训练（双卡 layer-split）

本机是 **RTX 3090 (24GB) + RTX 3060 (12GB)**。长序列不能靠 DDP「拼显存」；用 `device_map` **按层拆到两张卡**：

- 全部 **full_attention** 层 + 大部分层 → 3090
- 部分 **linear_attention** 层（28–30）→ 3060
- QLoRA 4-bit + gradient checkpointing + TRL chunked CE（已打补丁适配 PEFT/`device_map`）

实测（最长样本）：

| `max_length` | 双卡 QLoRA | 单卡 3090 QLoRA |
| --- | --- | --- |
| 16384 | 完整 epoch 默认（防碎片/偶发 OOM） | — |
| 20480 | 冒烟 OK，全量约 step 8 Triton OOM | OOM |
| 21504 | 单条探针 OK，全量易 OOM | OOM |
| 22528 / 32768 / 65536 | OOM | OOM |

因此默认 **`max_length=16384`**（双卡 layer-split + QLoRA；超长样本截断）。

**注意：** 不要设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。在本机（NVML 驱动不匹配）会触发 `PeerToPeerAccess` / `nvmlInit` 断言。

```bash
cd training/paper_trail
uv sync
chmod +x run_train.sh

# 模板检查
CUDA_VISIBLE_DEVICES=0,1 ./run_train.sh --dry-run \
  --config ../../configs/paper_trail/train_qlora_16k.toml

# 冒烟（1 step）
CUDA_VISIBLE_DEVICES=0,1 ./run_train.sh \
  --config ../../configs/paper_trail/train_qlora_16k.toml \
  --max-steps 1 --max-samples 4

# 完整 1 epoch（双卡 QLoRA，max_length=16384）
mkdir -p ../../data/paper_trail/models/qwen35-4b-sft-qlora-16k
CUDA_VISIBLE_DEVICES=0,1 ./run_train.sh \
  --config ../../configs/paper_trail/train_qlora_16k.toml \
  2>&1 | tee ../../data/paper_trail/models/qwen35-4b-sft-qlora-16k/train.log

# 中断后从最近 checkpoint 恢复（save_steps=50）
CUDA_VISIBLE_DEVICES=0,1 ./run_train.sh \
  --config ../../configs/paper_trail/train_qlora_16k.toml \
  --resume \
  2>&1 | tee -a ../../data/paper_trail/models/qwen35-4b-sft-qlora-16k/train.log
```

配置（推荐）：[`configs/paper_trail/train_qlora_16k.toml`](../../configs/paper_trail/train_qlora_16k.toml)  
（同内容别名：[`train_qlora_32k.toml`](../../configs/paper_trail/train_qlora_32k.toml)，文件名保留历史探测痕迹）

- 基座：`model/Qwen/Qwen3.5-4B`
- 输出：`data/paper_trail/models/qwen35-4b-sft-qlora-16k/`
- 产物：`adapter/` + `train_metrics.json`
- 给 vLLM 用：`uv run python merge_lora.py` → `.../merged/`

备选短上下文单卡 bf16 LoRA：[`configs/paper_trail/train.toml`](../../configs/paper_trail/train.toml)（`max_length=4096`）。

## 检查清单

- [x] `to_sft.py`：去 thinking、保上下文、预算内智能压缩
- [x] `apply_chat_template` tools / tool_call / tool_response
- [x] 双卡 QLoRA 冒烟（layer-split）；完整 epoch 默认 `max_length=16384`
- [x] 完整 1 epoch 写出最终 adapter + `train_metrics.json`（`train_loss≈0.557`，`epoch=1.0`）
- 校验：`./verify_artifacts.sh ../../data/paper_trail/models/qwen35-4b-sft-qlora-16k`
