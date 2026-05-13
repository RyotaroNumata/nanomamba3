#!/bin/bash

# Mamba-3 bilingual (English + Japanese) pretraining run.
# Single GPU (RTX 3090 or similar, 24GB VRAM).
#
# Differences from speedrun_mamba3.sh (English-only):
#   - NANOCHAT_JA_RATIO=0.3     : 30% Japanese, 70% English in training data
#   - Japanese shards downloaded : ~73 shards at JA_RATIO=0.3
#   - Tokenizer retrained        : bilingual BPE on mixed EN+JA text
#
# Differences from speedrun_bilingual.sh (GPT transformer):
#   - --model-arch mamba3        : Mamba-3 SSM instead of Transformer
#   - --device-batch-size=2      : VRAM limit (Triton SSD activations at T=2048)
#   - python (not torchrun)      : single GPU, grad_accum handles effective batch size
#
# Usage:
#   bash runs/speedrun_mamba3_bilingual.sh
#   WANDB_RUN=mamba3-bilingual bash runs/speedrun_mamba3_bilingual.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $NANOCHAT_BASE_DIR

# Japanese ratio: fraction of training data that is Japanese (default: 0.3 = 30%)
if [ -z "$NANOCHAT_JA_RATIO" ]; then
    export NANOCHAT_JA_RATIO=0.3
fi
echo "NANOCHAT_JA_RATIO = $NANOCHAT_JA_RATIO"

# Number of Japanese shards.
# At JA_RATIO=0.3 with 170 EN shards: 170 * 0.3 / 0.7 ≈ 73 JA shards.
JA_SHARDS=73

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
# Tokenizer + dataset (bilingual)

# Download initial EN shards for tokenizer training
python -m nanochat.dataset -n 8
# Download initial JA shards for tokenizer training
python -m nanochat.dataset -n 4 -l ja

# Kick off full EN + JA dataset downloads in background
python -m nanochat.dataset -n 170 &
EN_DOWNLOAD_PID=$!
python -m nanochat.dataset -n $JA_SHARDS -l ja &
JA_DOWNLOAD_PID=$!

# Train bilingual BPE tokenizer (NANOCHAT_JA_RATIO controls mix ratio)
# Skip if tokenizer already exists (delete $NANOCHAT_BASE_DIR/tokenizer.json to retrain)
if [ -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "Tokenizer already exists at $NANOCHAT_BASE_DIR/tokenizer/, skipping training"
else
    python -m scripts.tok_train
    python -m scripts.tok_eval
fi

# -----------------------------------------------------------------------------
# Base model pretraining (Mamba-3, bilingual)

echo "Waiting for dataset downloads to complete..."
wait $EN_DOWNLOAD_PID
wait $JA_DOWNLOAD_PID

# Mamba-3 d12 bilingual:
#   - NANOCHAT_JA_RATIO is inherited from env, picked up by dataloader automatically
#   - --device-batch-size=2: VRAM limit at T=2048 with Triton SSD kernel
#   - grad_accum auto-scales to match total_batch_size=524,288 tokens
python -m scripts.base_train \
    --model-arch mamba3 \
    --depth=12 \
    --target-param-data-ratio=12 \
    --device-batch-size=2 \
    --sample-every=-1 \
    --matrix-lr=0.001 \
    --run=$WANDB_RUN \
    --model-tag=mamba3-bilingual-d12 \
    "$@"

# Evaluate base model (use same tag as training, passed via "$@")
MODEL_TAG=$(python -c "
import sys
args = sys.argv[1:]
tag = 'mamba3-bilingual-d12'
for i, a in enumerate(args):
    if a.startswith('--model-tag='):
        tag = a.split('=', 1)[1]
    elif a == '--model-tag' and i+1 < len(args):
        tag = args[i+1]
print(tag)
" -- "$@")
python -m scripts.base_eval --device-batch-size=2 --model-tag=$MODEL_TAG

# -----------------------------------------------------------------------------
# Report

python -m nanochat.report generate
