# nanomamba3 — Build a Pure Mamba-3 Language Model from Scratch

[日本語](README.md) | **English**

This repository is a project for **building a bilingual (English + Japanese) language model using Mamba-3 from scratch**.

Built on top of nanochat, it covers every stage needed for an LLM — tokenizer training, pretraining, SFT (Supervised Fine-Tuning), evaluation, and inference — all in a single repository. The Mamba-3 model implementation is built from scratch in **pure PyTorch**, covering the SSM scan, MIMO rank-R state updates, and partial RoPE.

The language model built in this repository uses a **pure Mamba-3 architecture**.  
All you need is a single GPU with 24GB VRAM — give it a try and build your own Mamba-3 LM!



## Architecture: What is Mamba-3?

Mamba-3 is a sequence model based on State Space Models (SSMs). By recursively updating a fixed-size hidden state `h_t`, it achieves **O(1) inference memory**.

**MIMO (Multi-Input Multi-Output) rank-R** is an extension that expands the SSM from a scalar to a matrix, enabling the model to learn inter-channel correlations. This implementation uses rank=2.  
See [nanochat/mamba3.py](nanochat/mamba3.py) for the implementation.


## Model Weights

Pre-trained model weights are publicly available on HuggingFace:

**[kikyo0114/nanochat-mamba3-mimo-r2](https://huggingface.co/kikyo0114/nanochat-mamba3-mimo-r2)**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "kikyo0114/nanochat-mamba3-mimo-r2",
    local_dir="~/.cache/nanochat/chatsft_checkpoints/mamba3_mimo_r2_10k_sft",
)
```

---

## Quick Start

### Setup

```bash
git clone https://github.com/RyotaroNumata/nanomamba3.git
cd nanomamba3
uv sync
source .venv/bin/activate
```

### Mamba-3 Pretraining (Single GPU)
The command below runs everything in one go — from data download and tokenizer training to pretraining and SFT!

```bash
bash runs/speedrun_mamba3_bilingual_sft.sh
```

Internally, it runs in the following order:
1. Pretraining (~10,000 steps)
2. SFT (SmolTalk + Japanese instruction data)

### Running Each Stage Individually

#### Data Download

```bash
# English data (ClimbMix-400B)
python -m nanochat.dataset -n 170

# Japanese data (FineWeb-2-edu-japanese)
python -m nanochat.dataset -n 73 -l ja
```

#### Tokenizer Training

```bash
python -m scripts.tok_train
python -m scripts.tok_eval   # verify
```

#### Pretraining
Use `NANOCHAT_JA_RATIO` to control the Japanese data mixing ratio.
```bash
WANDB_RUN=mamba3_pretrain NANOCHAT_JA_RATIO=0.3 python -m scripts.base_train \
    --model-arch=mamba3 \
    --mamba-use-mimo \
    --depth=12 \
    --device-batch-size=2 \
    --matrix-lr=0.001 \
    --run=mamba3_pretrain \
    --model-tag=mamba3_mimo_r2

# Evaluate
python -m scripts.base_eval --device-batch-size=2 --model-tag=mamba3_mimo_r2
```

#### SFT

```bash
WANDB_RUN=mamba3_sft NANOCHAT_JA_RATIO=0.3 python -m scripts.chat_sft \
    --device-batch-size=8 \
    --model-tag=mamba3_mimo_r2 \
    --output-tag=mamba3_mimo_r2_sft \
    --run=mamba3_sft

# Evaluate
python -m scripts.chat_eval -i sft --model-tag=mamba3_mimo_r2_sft
```

### Chat
Run the commands below to chat with the model!
```bash
# CLI (temperature=1.0 recommended)
python -m scripts.chat_cli -i sft -g mamba3_mimo_r2_10k_sft -t 1.0 -p "Hello!"

# Web UI (open localhost:8000 in your browser)
python -m scripts.chat_web -g mamba3_mimo_r2_10k_sft -t 1.0
```

> **Note on temperature**: Due to the structural constraint of compressing context into a fixed-size SSM state, Mamba-3 tends to produce repetition loops at low temperatures. Using temperature=1.0 is recommended.

---

## File Structure

```
nanochat/
├── mamba3.py              # Mamba-3 MIMO SSM implementation (core of this project)
├── gpt.py                 # GPT Transformer (for comparison, runs on the same training loop)
├── engine.py              # Inference engine (KV cache / SSM recurrent inference)
├── tokenizer.py           # BPE tokenizer (32K vocab, bilingual EN/JA)
├── dataloader.py          # Distributed-aware data loader
├── dataset.py             # Data shard downloader
├── optim.py               # AdamW + Muon optimizer
├── checkpoint_manager.py  # Checkpoint save/load
├── common.py              # COMPUTE_DTYPE, DDP setup
└── NOTICE                 # Copyright and modification notice for mamba3.py

scripts/
├── base_train.py          # Pretraining main loop
├── base_eval.py           # Base model evaluation (CORE + BPB)
├── chat_sft.py            # SFT training
├── chat_eval.py           # Chat model evaluation
├── chat_cli.py            # CLI chat
└── chat_web.py            # Web UI chat

runs/
├── speedrun_mamba3_bilingual_sft.sh  # Mamba-3 full pipeline (recommended)
├── speedrun_mamba3_bilingual.sh      # Mamba-3 pretraining only
└── speedrun.sh                       # GPT-2 speedrun (Transformer version)
```

---

## Training Data

### Pretraining Data

- **English**: [ClimbMix-400B](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
- **Japanese**: [FineWeb-2 Japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese) (derived from [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), licensed under [ODC-By v1.0](https://opendatacommons.org/licenses/by/1.0/))
  - Source web data: [CommonCrawl](https://commoncrawl.org) ([Terms of Use](https://commoncrawl.org/terms-of-use))

**Attribution (ODC-By v1.0 requirement)**: When publishing any artifact that uses FineWeb-2 Japanese, attribution to FineWeb2 and CommonCrawl is required.

### SFT Data

- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) (English)
- [OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) (Japanese conversation)
- [Dolly-JA](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja)
- [Magpie-JA](https://huggingface.co/datasets/Aratako/Magpie-Qwen2.5-72B-Instruct-Japanese-300K-Filtered)


## License

MIT

[nanochat/mamba3.py](nanochat/mamba3.py) is derived from [mamba3-minimal](https://github.com/VikramLex/mamba3-minimal) (Copyright 2026 Vikram Karlex) and is provided under the **Apache License 2.0**. See [nanochat/NOTICE](nanochat/NOTICE) for full attribution and modification details.

---

## Acknowledgements

- [nanochat](https://github.com/karpathy/nanochat) — LLM training framework by Andrej Karpathy
- [mamba3-minimal](https://github.com/VikramLex/mamba3-minimal) — base implementation for Mamba-3
