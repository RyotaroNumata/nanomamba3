---
license: apache-2.0
language:
  - en
  - ja
tags:
  - mamba
  - ssm
  - causal-lm
  - bilingual
  - onnx
---

# nanochat-mamba3-mimo-r2-sft

A bilingual (English + Japanese) chat model based on a **from-scratch pure-PyTorch implementation of Mamba-3 MIMO rank-2**, trained end-to-end using the [nanochat](https://github.com/RyotaroNumata/nanomamba3) framework.

No Triton, no `mamba_ssm` — the SSM scan (`ssd_siso` / `ssd_mimo`), data-dependent decay, MIMO rank-R state updates, and partial RoPE are all implemented in plain PyTorch. See [`nanochat/mamba3.py`](https://github.com/RyotaroNumata/nanomamba3/blob/master/nanochat/mamba3.py) for the implementation.

## Model details

| Item | Value |
|------|-------|
| Architecture | Mamba-3 MIMO rank-2 |
| Parameters | ~113M |
| Layers | 12 |
| Hidden size | 768 |
| MIMO rank | 2 |
| Chunk size | 32 |
| Vocabulary | 32,768 (BPE, GPT-4 split pattern) |
| Languages | English + Japanese (JA ratio 30%) |
| Pretraining data | NVIDIA ClimbMix-400B (EN) + FineWeb-2-edu-ja (JA) |
| SFT data | SmolTalk + Dolly-ja + OASST2-ja + Magpie-ja |
| Pretraining steps | 10,000 |
| SFT steps | 10,000 |
| Hardware | Single NVIDIA RTX 3090 (24 GB) |

## Usage

### Setup

```bash
git clone https://github.com/RyotaroNumata/nanomamba3.git
cd nanochat
uv sync
source .venv/bin/activate
```

### Download weights

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "kikyo0114/nanochat-mamba3-mimo-r2",
    local_dir="~/.cache/nanochat/chatsft_checkpoints/mamba3_mimo_r2_10k_sft",
)
```

### Chat (CLI)

```bash
# temperature=1.0 recommended for Mamba-3
python -m scripts.chat_cli -i sft -g mamba3_mimo_r2_10k_sft -t 1.0 -p "Why is the sky blue?"
python -m scripts.chat_cli -i sft -g mamba3_mimo_r2_10k_sft -t 1.0 -p "日本語でも話せますか？"

# Interactive mode
python -m scripts.chat_cli -i sft -g mamba3_mimo_r2_10k_sft -t 1.0
```

### Chat (Web UI)

```bash
python -m scripts.chat_web
```

### ONNX inference (CPU, no GPU required)

```bash
# Export
python -m scripts.export_onnx \
    --model-tag mamba3_mimo_r2_10k_sft \
    --fp32 --output /tmp/mamba3_step_fp32.onnx --export-prefill

# Quantize to int8
python -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('/tmp/mamba3_step_fp32.onnx',         '/tmp/mamba3_step_int8.onnx',         weight_type=QuantType.QInt8)
quantize_dynamic('/tmp/mamba3_step_fp32_prefill.onnx', '/tmp/mamba3_step_int8_prefill.onnx', weight_type=QuantType.QInt8)
"

# Run (int8 + chunk prefill = fastest CPU config)
python -m scripts.chat_onnx \
    --onnx /tmp/mamba3_step_int8.onnx \
    --onnx-prefill /tmp/mamba3_step_int8_prefill.onnx \
    -p "Why is the sky blue?"
```

## Evaluation scores (SFT, d12, 10k steps)

| Task | GPT Transformer (bilingual_v2) | **Mamba-3 MIMO rank-2** |
|------|-------------------------------|------------------------|
| ARC-Easy | 36.45% | 33.50% |
| ARC-Challenge | 33.28% | 28.84% |
| MMLU | 31.89% | 30.25% |
| GSM8K | 5.00% | 1.06% |
| HumanEval | 9.15% | 0.61% |
| SpellingBee | 99.22% | 82.81% |
| JCommonsenseQA | 35.48% | 33.24% |
| Base CORE | — | 0.1122 |
| ChatCORE | — | 0.1710 |

## Inference speed (RTX 3090, 100 tokens generated, averaged over 3 runs)

| Backend | prompt=32 | prompt=128 | prompt=512 | Notes |
|---------|-----------|------------|------------|-------|
| CUDA (PyTorch) | 51.1 tok/s | 49.1 tok/s | 51.1 tok/s | **Constant** regardless of prompt length (O(1) recurrent inference) |
| CPU (PyTorch) | 21.9 tok/s | 20.8 tok/s | 17.7 tok/s | |
| ONNX fp32 (CPU) | 32.1 tok/s | 18.6 tok/s | 6.9 tok/s | Sequential per-token prefill |
| ONNX fp32 + chunk prefill (CPU) | 38.6 tok/s | 31.7 tok/s | 18.4 tok/s | 1.21×/1.71×/2.64× speedup |
| ONNX int8 (CPU) | 55.6 tok/s | 32.1 tok/s | 11.8 tok/s | |
| **ONNX int8 + chunk prefill (CPU)** | **64.6 tok/s** | **48.3 tok/s** | **23.6 tok/s** | Fastest CPU config; beats CUDA PyTorch at short prompts |

> The O(1) memory of SSMs means Mamba-3 holds its decode speed as context grows, while Transformers slow down due to KV cache growth.

## Limitations

- **Small model (~113M params, d12)**: Comparable to GPT-1 in scale. Expect kindergartener-level reasoning — good for creative tasks, unreliable for math/code.
- **Mamba-3 is still experimental**: This is a research implementation. The architecture differs from the official `mamba_ssm` in several ways (data-dependent A decay, MIMO rank-2, partial RoPE).
- **Temperature sensitivity**: Use `temperature=1.0`. Lower values (< 0.5) cause repetition loops due to SSM state accumulation.
- **Japanese quality**: JA ratio was 30% during training. Japanese fluency is functional but limited compared to dedicated Japanese models.

## Architecture differences from mamba3-minimal

`nanochat/mamba3.py` is derived from [mamba3-minimal](https://github.com/VikramLex/mamba3-minimal) with the following modifications:

| Item | Original | nanochat |
|------|----------|----------|
| A decay | Fixed `A_log` parameter | Data-dependent `dd_A` (per-token, per-head) |
| State update | SISO (rank-1) | MIMO rank-R outer-product updates |
| Position encoding | None | Partial RoPE on a subset of heads |
| B/C bias init | Default | Zero-initialized for stable early training |
| SSD scan precision | Model dtype | float32 (numerical stability) |

See [nanochat/NOTICE](https://github.com/RyotaroNumata/nanomamba3/blob/master/nanochat/NOTICE) for full attribution.

## License

`nanochat/mamba3.py` (and this model) is licensed under the **Apache License 2.0**, derived from [mamba3-minimal](https://github.com/VikramLex/mamba3-minimal) (Copyright 2026 Vikram Karlex).

The rest of the nanochat codebase is MIT licensed.

## Citation

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title  = {nanochat: The best ChatGPT that $100 can buy},
  year   = {2025},
  url    = {https://github.com/karpathy/nanochat}
}
```
