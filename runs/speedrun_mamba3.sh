#!/bin/bash

# Mamba-3 pretraining run — single GPU (RTX 3090 or similar, 24GB VRAM).
# Trains a Mamba-3 d12 model (~153M params) and compares against the GPT-2 baseline.
#
# Key differences from speedrun.sh (GPT transformer):
#   - --model-arch mamba3            : Mamba-3 SSM instead of Transformer
#   - --device-batch-size=2          : VRAM limit (SSD activations ~11GB at B=2, T=2048)
#   - torch.compile is skipped       : mamba-ssm already provides optimised CUDA kernels
#   - No --nproc_per_node > 1 needed : single GPU run
#
# Usage:
#   bash runs/speedrun_mamba3.sh
#   WANDB_RUN=mamba3-d12 screen -L -Logfile runs/speedrun_mamba3.log -S mamba3 bash runs/speedrun_mamba3.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
# Reduces VRAM fragmentation (recommended for large activations)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Report header

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer + dataset

python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model pretraining (Mamba-3)

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Mamba-3 d12: device_batch_size=2 (VRAM limit at T=2048), grad_accum auto-scales
# to match the same total_batch_size as GPT d12 (524,288 tokens).
# --target-param-data-ratio=8 matches the GPT speedrun setting.
python -m scripts.base_train \
    --model-arch mamba3 \
    --depth=12 \
    --target-param-data-ratio=8 \
    --device-batch-size=2 \
    --run=$WANDB_RUN \
    --model-tag=mamba3-d12

# Evaluate base model
python -m scripts.base_eval --device-batch-size=2 --model-tag=mamba3-d12

# -----------------------------------------------------------------------------
# Report

python -m nanochat.report generate
