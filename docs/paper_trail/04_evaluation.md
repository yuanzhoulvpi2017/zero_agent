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
uv run python -m run_paper_trail --sft
```

访问 http://127.0.0.1:8000。该模式读 `agent_sft.toml`，**不需要** DeepSeek 密钥。采集蒸馏仍用 `uv run python -m run_paper_trail` + `agent.toml`。

## 检查清单

- [x] 合并 LoRA 为 vLLM 可加载权重
- [x] vLLM OpenAI 服务脚本（CUDA 预热 + 64k + qwen3 reasoning + qwen3_coder tools）
- [x] 网页 `--sft` 指向本地推理
- [ ] 与商业模型 / 未训基座的批量对比评测
