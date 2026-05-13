#!/bin/bash
set -euo pipefail

# Mamba-3 MIMO rank-2 bilingual speedrun: pretraining + SFT (English + Japanese).
# Single GPU (RTX 3090 or similar, 24GB VRAM).
# Expected total time: ~4–5 hours at 10k pretraining steps + SFT.
#
# Key differences from speedrun_bilingual_sft.sh (GPT transformer):
#   - --model-arch mamba3 --mamba-use-mimo  : Mamba-3 MIMO rank-2 SSM
#   - --device-batch-size=2                 : VRAM limit at T=2048
#   - python (not torchrun)                 : single GPU, grad_accum handles batch
#   - --matrix-lr=0.001                     : tuned LR for Mamba-3
#   - No torch.compile                      : incompatible with SSM scan ops
#
# Usage:
#   bash runs/speedrun_mamba3_bilingual_sft.sh
#   MODEL_TAG=mamba3_mimo_r2 bash runs/speedrun_mamba3_bilingual_sft.sh
#   MODEL_TAG=mamba3_mimo_r2 screen -L -Logfile runs/mamba3_bilingual_sft.log \
#       -S mamba3sft bash runs/speedrun_mamba3_bilingual_sft.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $NANOCHAT_BASE_DIR

# MODEL_TAG: base model checkpoint tag + wandb base run name.
# The SFT model tag will be ${MODEL_TAG}_sft.
if [ -z "${MODEL_TAG:-}" ]; then
    MODEL_TAG="mamba3_mimo_r2"
fi
SFT_TAG="${MODEL_TAG}_sft"
echo "MODEL_TAG     = $MODEL_TAG"
echo "SFT_TAG       = $SFT_TAG"

# Japanese data ratio: 30% JA / 70% EN (matches bilingual_v2 baseline)
if [ -z "${NANOCHAT_JA_RATIO:-}" ]; then
    export NANOCHAT_JA_RATIO=0.3
fi
echo "NANOCHAT_JA_RATIO = $NANOCHAT_JA_RATIO"

# Number of Japanese shards: 170 EN shards * (JA_RATIO / (1 - JA_RATIO)) ≈ 73
JA_SHARDS=73

# -----------------------------------------------------------------------------
# Python venv setup

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb
# Set WANDB_RUN before running to enable logging, e.g.:
#   WANDB_RUN=${MODEL_TAG} bash runs/speedrun_mamba3_bilingual_sft.sh
if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Report header

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer + dataset (bilingual)

# Download initial shards for tokenizer training
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 4 -l ja

# Kick off full downloads in background while tokenizer trains
python -m nanochat.dataset -n 170 &
EN_DOWNLOAD_PID=$!
python -m nanochat.dataset -n $JA_SHARDS -l ja &
JA_DOWNLOAD_PID=$!

# Train bilingual BPE tokenizer (skip if already trained)
if [ -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "Tokenizer already exists, skipping training."
else
    python -m scripts.tok_train
    python -m scripts.tok_eval
fi

# -----------------------------------------------------------------------------
# Base model pretraining (Mamba-3 MIMO rank-2, bilingual)

echo "Waiting for dataset downloads to complete..."
wait $EN_DOWNLOAD_PID
wait $JA_DOWNLOAD_PID

# Mamba-3 MIMO rank-2 d12:
#   --mamba-use-mimo               : enable MIMO rank-2 (default rank=2)
#   --device-batch-size=2          : VRAM limit at T=2048
#   --target-param-data-ratio=12   : compute-optimal D:N ratio for this scale
#   --matrix-lr=0.001              : tuned for Mamba-3 (vs 0.003 for Transformer)
#   --sample-every=-1              : skip text samples (saves time)
WANDB_RUN=${WANDB_RUN} python -m scripts.base_train \
    --model-arch=mamba3 \
    --mamba-use-mimo \
    --depth=12 \
    --target-param-data-ratio=12 \
    --device-batch-size=2 \
    --matrix-lr=0.001 \
    --sample-every=-1 \
    --run=${WANDB_RUN} \
    --model-tag=${MODEL_TAG}

# Evaluate base model
python -m scripts.base_eval --device-batch-size=2 --model-tag=${MODEL_TAG}

# -----------------------------------------------------------------------------
# SFT (Mamba-3 MIMO rank-2, bilingual)

# Download synthetic identity conversations (nanochat personality)
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Run SFT.
# NANOCHAT_JA_RATIO > 0 automatically adds JapaneseInstruct and
# Japanese identity conversations to the training mixture.
WANDB_RUN=${SFT_TAG} NANOCHAT_JA_RATIO=${NANOCHAT_JA_RATIO} python -m scripts.chat_sft \
    --device-batch-size=8 \
    --model-tag=${MODEL_TAG} \
    --output-tag=${SFT_TAG} \
    --run=${SFT_TAG}

# Evaluate SFT model
python -m scripts.chat_eval -i sft --model-tag=${SFT_TAG}

# Chat with the model (uncomment to try):
# python -m scripts.chat_cli -i sft -g ${SFT_TAG} -t 1.0 -p "こんにちは！自己紹介してください。"
# python -m scripts.chat_cli -i sft -g ${SFT_TAG} -t 1.0 -p "Why is the sky blue?"
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Report

python -m nanochat.report generate
