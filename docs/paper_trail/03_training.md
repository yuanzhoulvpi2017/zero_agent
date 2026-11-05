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

## Loss mask 教学示例（仅分词）

- [x] 转换与全量 mask 检查已合并到 [code002_to_sft.py](../../agent/paper_trail/simple/code002_to_sft.py)，不再保留独立 code003。运行命令、全量输出索引见 [02_dataset.md](02_dataset.md#simple-日志转换教学示例)。

示例使用模型原始 `tokenizer.apply_chat_template`，不手工修改模板、不切换 TRL 模板、不依赖 generation 标记。两次调用都传入完整 tools 和 `enable_thinking=False`：

1. 编码完整 messages，设置 `add_generation_prompt=False`。
2. 编码 `messages[:-1]`，设置 `add_generation_prompt=True` 得到 prompt。
3. 逐个检查 prompt token 是否等于完整 token 序列的前缀；不匹配直接报错。
4. prompt 部分 labels 设为 -100，只监督剩余的本次输出。

```python
assert full_ids[:len(prefix_ids)] == prefix_ids
labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
loss_mask = [int(label != -100) for label in labels]
```

| 内容 | loss_mask | labels |
| --- | --- | --- |
| 工具 schema、system、user、工具返回、历史 assistant | 0 | -100 |
| 原生 assistant 生成前缀，包括空 think | 0 | -100 |
| 本次 tool call 的工具名、参数和格式 | 1 | 原 token ID |
| 本次无 tool call 的正文回复 | 1 | 原 token ID |
| 本次 assistant 结束标记 | 1 | 原 token ID |

有 tool_calls 的 assistant，其正文在 SFT 转换时清空，历史与本次输出一致。每份日志只监督最后一条 assistant；历史输出在对应的其他调用样本里监督。工具返回保留上下文角色，Qwen 将其渲染为 user / tool_response 包装，mask 仍为 0。

**think 边界统一：** 本地原始模板的空包装实际为 `<think>\n\n</think>\n\n`，保留原生换行和标签。它已经由 prompt 给出，所以当前所有样本的空 think 前缀都在 mask=0 部分。旧输出中出现 think 在 mask=1，是此前 TRL 全 assistant 标记版本的结果；紧凑 `<think></think>` 是此前手工模板版本的结果。这两种旧目录均已停止作为检查依据；它们的边界不能与当前批次混用。当前没有修改模型模板，也没有移除原生格式 token。

`attention_mask` 仍为 1，上下文继续参与模型计算。手算 causal loss 时，用 `logits[:, :-1]` 对齐 `labels[:, 1:]`，忽略 -100 并按有效目标数平均。传给 causal LM 的 labels 保持原长度，由模型内部移位。

生产 `train.py` 仍使用 `assistant_only_loss=True`，会监督所有 assistant；这与本示例只监督本次输出不同。本次没有启动或修改生产训练。后续接入训练须使用已生成 labels 或等效的 prompt/completion 处理，不能重新套用全 assistant mask。

仅运行相关回归检查：

```bash
training/paper_trail/.venv/bin/python training/paper_trail/check_loss_mask.py
```

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
