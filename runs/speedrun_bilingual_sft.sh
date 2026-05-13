#!/bin/bash
set -euo pipefail  # exit on error, treat unset vars as errors, propagate pipe failures

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun_bilingual_sft.sh
# 2) With a named tag (sets checkpoint dir + wandb run name):
# MODEL_TAG=bilingual_v2 bash runs/speedrun_bilingual_sft.sh
# 3) In a screen session:
# MODEL_TAG=bilingual_v2 screen -L -Logfile runs/sft.log -S sft bash runs/speedrun_bilingual_sft.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# MODEL_TAG: controls both the checkpoint directory and the wandb run name.
#   - Checkpoint saved to: ~/.cache/nanochat/chatsft_checkpoints/<MODEL_TAG>/
#   - wandb run name     : <MODEL_TAG>  (set to "dummy" to disable wandb)
# Usage:
#   MODEL_TAG=bilingual_v2 bash runs/speedrun_bilingual_sft.sh
if [ -z "${MODEL_TAG:-}" ]; then
    MODEL_TAG="d12_bilingual_sft"
fi
echo "MODEL_TAG = $MODEL_TAG"

# Japanese ratio: must match the value used in pretraining so the tokenizer
# and model are aligned. Defaults to 0.3 (= 30% Japanese).
if [ -z "${NANOCHAT_JA_RATIO:-}" ]; then
    export NANOCHAT_JA_RATIO=0.3
fi
echo "NANOCHAT_JA_RATIO = $NANOCHAT_JA_RATIO"

# -----------------------------------------------------------------------------
# SFT (supervised fine-tuning)

# Run SFT.
# When NANOCHAT_JA_RATIO > 0, JapaneseInstruct and Japanese identity conversations
# are automatically added to the training mixture alongside the English tasks.
torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
    --device-batch-size=8 \
    --output-tag=$MODEL_TAG \
    --run=$MODEL_TAG

# Evaluate the chat model
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft --model-tag=$MODEL_TAG

# Chat with the model (uncomment to try):
# python -m scripts.chat_cli -p "こんにちは！自己紹介してください。"
# python -m scripts.chat_cli -p "Why is the sky blue?"
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report

python -m nanochat.report generate
