#!/bin/bash

# Bilingual (English + Japanese) training speedrun.
# Follows the same structure as runs/speedrun.sh, with NANOCHAT_JA_RATIO added
# to mix Japanese data into every stage of the pipeline.
#
# 1) Example launch (simplest):
#    bash runs/speedrun_bilingual.sh
# 2) In a screen session (run takes a while):
#    screen -L -Logfile runs/speedrun_bilingual.log -S speedrun_bilingual bash runs/speedrun_bilingual.sh
# 3) With wandb logging:
#    WANDB_RUN=bilingual screen -L -Logfile runs/speedrun_bilingual.log -S speedrun_bilingual bash runs/speedrun_bilingual.sh
# 4) With a custom JA ratio (default: 0.2 = 20% Japanese):
#    NANOCHAT_JA_RATIO=0.3 bash runs/speedrun_bilingual.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Japanese ratio: fraction of training data that is Japanese (default: 0.2 = 20%)
# Propagated to tok_train, base_train, and chat_sft automatically.
if [ -z "$NANOCHAT_JA_RATIO" ]; then
    export NANOCHAT_JA_RATIO=0.3
fi
echo "NANOCHAT_JA_RATIO = $NANOCHAT_JA_RATIO"

# Number of Japanese shards to download.
# At JA_RATIO=0.3 with ~170 EN shards: 170 * 0.3 / 0.7 ≈ 73 JA shards.
# Adjust proportionally if you change NANOCHAT_JA_RATIO or EN shard count.
JA_SHARDS=73

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Clear and start the report

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of English data (8 shards) for tokenizer training
python -m nanochat.dataset -n 8

# Download the first batch of Japanese data (proportional to JA_RATIO) for tokenizer training
# At NANOCHAT_JA_RATIO=0.2: ~8 * 0.2/0.8 = 2 shards needed; download a few more for safety
python -m nanochat.dataset -n 4 -l ja

# Kick off background downloads of the full EN and JA datasets while tokenizer trains
python -m nanochat.dataset -n 170 &
EN_DOWNLOAD_PID=$!
python -m nanochat.dataset -n $JA_SHARDS -l ja &
JA_DOWNLOAD_PID=$!

# Train the bilingual BPE tokenizer on mixed EN+JA text.
# NANOCHAT_JA_RATIO controls the mix ratio; --max-chars default auto-adjusts to 500M.
python -m scripts.tok_train

# Evaluate the tokenizer (reports compression ratio for English and Japanese)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

echo "Waiting for dataset downloads to complete..."
wait $EN_DOWNLOAD_PID
wait $JA_DOWNLOAD_PID

# Pretrain with bilingual data.
# NANOCHAT_JA_RATIO is inherited from the environment and picked up automatically
# by the dataloader to interleave EN and JA parquet files.
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 \
    --target-param-data-ratio=8 \
    --device-batch-size=8 \
    --run=$WANDB_RUN \
    --resume-from-step=5000 \
    --num-iterations=12000

# Evaluate the base model (CORE score, BPB, sample generation)
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- --device-batch-size=8

# -----------------------------------------------------------------------------
# SFT (supervised fine-tuning)

# Download English identity conversations
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \S
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Run SFT.
# When NANOCHAT_JA_RATIO > 0, JapaneseInstruct and Japanese identity conversations
# are automatically added to the training mixture alongside the English tasks.
torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
    --device-batch-size=8 \
    --run=$WANDB_RUN

# Evaluate the chat model
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft

# Chat with the model (uncomment to try):
# python -m scripts.chat_cli -p "こんにちは！自己紹介してください。"
# python -m scripts.chat_cli -p "Why is the sky blue?"
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report

python -m nanochat.report generate
