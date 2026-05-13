# nanomamba3

**English** | [日本語](README.md)

> **This fork implements Mamba-3 MIMO from scratch in pure PyTorch — no Triton, no mamba_ssm — and runs the full pipeline from pretraining to bilingual SFT on a single GPU.**
> We built our own SSM scan (`ssd_siso` / `ssd_mimo`), added data-dependent decay, MIMO rank-R state updates, and partial RoPE on top of the [Mamba-3 paper](https://arxiv.org/abs/2603.15569). The result is a single-file drop-in that plugs into the nanochat training loop alongside the GPT Transformer, complete with ONNX export for on-device inference. See [nanochat/mamba3.py](nanochat/mamba3.py) and [runs/speedrun_mamba3_bilingual_sft.sh](runs/speedrun_mamba3_bilingual_sft.sh).

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat is the simplest experimental harness for training LLMs. It is designed to run on a single GPU node, the code is minimal/hackable, and it covers all major LLM stages including tokenization, pretraining, finetuning, evaluation, inference, and a chat UI. For example, you can train your own GPT-2 capability LLM (which cost ~$43,000 to train in 2019) for only $48 (~2 hours of 8XH100 GPU node) and then talk to it in a familiar ChatGPT-like web UI. On a spot instance, the total cost can be closer to ~$15. More generally, nanochat is configured out of the box to train an entire miniseries of compute-optimal models by setting one single complexity dial: `--depth`, the number of layers in the GPT transformer model (GPT-2 capability happens to be approximately depth 26). All other hyperparameters (the width of the transformer, number of heads, learning rate adjustments, training horizons, weight decays, ...) are calculated automatically in an optimal way.

For questions about the repo, I recommend either using [DeepWiki](https://deepwiki.com/karpathy/nanochat) from Devin/Cognition to ask questions about the repo, or use the [Discussions tab](https://github.com/karpathy/nanochat/discussions), or come by the [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) channel on Discord.

## Time-to-GPT-2 Leaderboard

Presently, the main focus of development is on tuning the pretraining stage, which takes the most amount of compute. Inspired by the modded-nanogpt repo and to incentivise progress and community collaboration, nanochat maintains a leaderboard for a "GPT-2 speedrun", which is the wall-clock time required to train a nanochat model to GPT-2 grade capability, as measured by the DCLM CORE score. The [runs/speedrun.sh](runs/speedrun.sh) script always reflects the reference way to train GPT-2 grade model and talk to it. The current leaderboard looks as follows:

| # | time | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 hours | - | 0.2565 | Original OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | bump total batch size to 1M tokens | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | change dataset to NVIDIA ClimbMix | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [round 1](https://x.com/karpathy/status/2031135152349524125) | Mar 9 2026 | 6ed7d1d | @karpathy |
| 5 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

The primary metric we care about is "time to GPT-2" - the wall clock time needed to outperform the GPT-2 (1.6B) CORE metric on an 8XH100 GPU node. The GPT-2 CORE score is 0.256525. In 2019, the training of GPT-2 cost approximately $43,000 so it is incredible that due to many advances over 7 years across the stack, we can now do so much faster and for well below $100 (e.g. at the current ~$3/GPU/hr, an 8XH100 node is ~$24/hr, so 2 hours is ~$48).

See [dev/LEADERBOARD.md](dev/LEADERBOARD.md) for more docs on how to interpret and contribute to the leaderboard.

## Getting started

### Reproduce and talk to GPT-2

The most fun you can have is to train your own GPT-2 and talk to it. The entire pipeline to do so is contained in the single file [runs/speedrun.sh](runs/speedrun.sh), which is designed to be run on an 8XH100 GPU node. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash runs/speedrun.sh
```

You may wish to do so in a screen session as this will take ~3 hours to run. Once it's done, you can talk to it via the ChatGPT-like web UI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and serve it:

```bash
python -m scripts.chat_web
```

And then visit the URL shown. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc. Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :).

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

A few more notes:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device_batch_size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't personally exercised all of these code paths so there might be sharp edges.

## Mamba-3 MIMO (SSM architecture)

This fork adds a from-scratch pure-PyTorch implementation of Mamba-3 MIMO rank-2, trained end-to-end from pretraining to bilingual (English + Japanese) SFT on a single GPU. See [nanochat/mamba3.py](nanochat/mamba3.py) for the implementation and [nanochat/NOTICE](nanochat/NOTICE) for attribution to the upstream [mamba3-minimal](https://github.com/VikramLex/mamba3-minimal).

```bash
# Full pipeline: pretraining → bilingual SFT (single GPU, ~4–5 hrs)
bash runs/speedrun_mamba3_bilingual_sft.sh

# Chat with the result (temperature=1.0 recommended for Mamba-3)
python -m scripts.chat_cli -i sft -g mamba3_mimo_r2_sft -t 1.0 -p "Why is the sky blue?"
```

### Eval scores (SFT, d12, 10k steps)

| Task | GPT Transformer (bilingual_v2) | Mamba-3 MIMO rank-2 |
|------|-------------------------------|---------------------|
| ARC-Easy | **36.45%** | 33.50% |
| ARC-Challenge | **33.28%** | 28.84% |
| MMLU | **31.89%** | 30.25% |
| GSM8K | **5.00%** | 1.06% |
| HumanEval | **9.15%** | 0.61% |
| SpellingBee | **99.22%** | 82.81% |
| JCommonsenseQA | **35.48%** | 33.24% |
| Base CORE | — | 0.1122 |
| ChatCORE | — | 0.1710 |

### Inference speed (RTX 3090, 100 tokens generated, averaged over 3 runs)

| Backend | prompt=32 | prompt=128 | prompt=512 | Notes |
|---------|-----------|------------|------------|-------|
| Mamba-3 MIMO — CUDA (PyTorch) | 51.1 tok/s | 49.1 tok/s | 51.1 tok/s | Speed is **constant** regardless of prompt length (O(1) recurrent inference) |
| GPT Transformer — CUDA (PyTorch) | 97.7 tok/s | 99.6 tok/s | 81.6 tok/s | Faster at short prompts; degrades at long context (KV cache growth) |
| Mamba-3 MIMO — CPU (PyTorch) | 21.9 tok/s | 20.8 tok/s | 17.7 tok/s | |
| Mamba-3 — ONNX fp32 (CPU) | 32.1 tok/s | 18.6 tok/s | 6.9 tok/s | Sequential per-token prefill |
| Mamba-3 — ONNX fp32 + chunk prefill (CPU) | 38.6 tok/s | 31.7 tok/s | 18.4 tok/s | chunk_size=32; 1.21×/1.71×/2.64× speedup vs sequential |
| Mamba-3 — ONNX int8 (CPU) | 55.6 tok/s | 32.1 tok/s | 11.8 tok/s | ~1.7× over fp32 |
| Mamba-3 — ONNX int8 + chunk prefill (CPU) | **64.6 tok/s** | **48.3 tok/s** | **23.6 tok/s** | chunk_size=32; 1.17×/1.50×/2.00× speedup vs sequential; **fastest across all CPU configs** |

> The O(1) memory property of SSMs means Mamba-3 holds its decode speed as context grows, while the Transformer slows down due to KV cache. At prompt=512, Mamba-3 CUDA (51 tok/s) is already closing the gap with the Transformer (81 tok/s).
> Chunk prefill amortises the SSD scan cost over 32-token blocks — the longer the prompt, the larger the speedup (2.64× fp32 / 2.00× int8 at 512 tokens).
> Notably, int8 + chunk prefill at short prompts (64.6 tok/s) beats CUDA PyTorch (51.1 tok/s).
> Run `python inference_bench.py --no-pytorch` to reproduce the ONNX numbers.

### ONNX export and on-device inference

Export a trained Mamba-3 SFT model to ONNX for CPU / mobile deployment:

```bash
# Export fp32 decode-step model + chunk-prefill model (recommended)
python -m scripts.export_onnx \
    --model-tag mamba3_mimo_r2_10k_sft \
    --fp32 \
    --output /tmp/mamba3_step_fp32.onnx \
    --export-prefill \
    --verify

# Quantize to int8 (decode-step and prefill separately)
python -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('/tmp/mamba3_step_fp32.onnx',        '/tmp/mamba3_step_int8.onnx',        weight_type=QuantType.QInt8)
quantize_dynamic('/tmp/mamba3_step_fp32_prefill.onnx', '/tmp/mamba3_step_int8_prefill.onnx', weight_type=QuantType.QInt8)
"

# Chat via ONNX (int8 + chunk prefill = fastest CPU config)
python -m scripts.chat_onnx \
    --onnx /tmp/mamba3_step_int8.onnx \
    --onnx-prefill /tmp/mamba3_step_int8_prefill.onnx \
    -p "Why is the sky blue?"

# Interactive mode
python -m scripts.chat_onnx \
    --onnx /tmp/mamba3_step_int8.onnx \
    --onnx-prefill /tmp/mamba3_step_int8_prefill.onnx
```

File sizes after export: fp32 decode ~528 MB, fp32 prefill ~530 MB, int8 decode ~133 MB, int8 prefill ~135 MB.

## Research

If you are a researcher and wish to help improve nanochat, two scripts of interest are [runs/scaling_laws.sh](runs/scaling_laws.sh) and [runs/miniseries.sh](runs/miniseries.sh). See [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) for related documentation. For quick experimentation (~5 min pretraining runs) my favorite scale is to train a 12-layer model (GPT-1 sized), e.g. like this:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

This uses wandb (run name "d12"), only runs the CORE metric on last step, and it doesn't sample and save intermediate checkpoints. I like to change something in the code, re-run a d12 (or a d16 etc) and see if it helped, in an iteration loop. To see if a run helps, I like to monitor the wandb plots for:

1. `val_bpb` (validation loss in vocab-size-invariant units of bits per byte) as a function of `step`, `total_training_time` and `total_training_flops`.
2. `core_metric` (the DCLM CORE socre)
3. VRAM utilization, `train/mfu` (Model FLOPS utilization), `train/tok_per_sec` (training throughput)

See an example [here](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044).

The important thing to note is that nanochat is written and configured around one single dial of complexity - the depth of the transformer. This single integer automatically determines all other hyperparameters (the width of the transformer, number of heads, learning rate adjustments, training horizons, weight decays, ...) so that the trained model comes out compute optimal. The idea is that the user doesn't have to think about or set any of this, they are simply asking for a smaller or bigger model using `--depth`, and everything "just works". By sweeping out the depth, you achieve the nanochat miniseries of compute optimal models at various sizes. GPT-2 capability model (which is of most interest at the moment) happens to be somewhere around d24-d26 range with the current code. But any candidate changes to the repo have to be principled enough that they work for all settings of depth.

## Running on CPU / MPS

The script [runs/runcpu.sh](runs/runcpu.sh) shows a very simple example of running on CPU or Apple Silicon. It dramatically shrinks the LLM that is being trained to make things fit into a reasonable time interval of a few ten minutes of training. You will not get strong results in this way.

## Precision / dtype

nanochat does not use `torch.amp.autocast`. Instead, precision is managed explicitly through a single global `COMPUTE_DTYPE` (defined in `nanochat/common.py`). By default this is auto-detected based on your hardware:

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | No reduced-precision tensor cores |

You can override the default with the `NANOCHAT_DTYPE` environment variable:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # force fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # force bf16
```

How it works: model weights are stored in fp32 (for optimizer precision), but our custom `Linear` layer casts them to `COMPUTE_DTYPE` during the forward pass. Embeddings are stored directly in `COMPUTE_DTYPE` to save memory. This gives us the same mixed-precision benefit as autocast but with full explicit control over what runs in which precision.

Note: `float16` training automatically enables a `GradScaler` in `base_train.py` to prevent gradient underflow. SFT suppors this too but RL currently does not. Inference in fp16 works fine everywhere.

## Guides

I've published a number of guides that might contain helpful information, most recent to least recent:

- [Feb 1 2026: Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) documents the first nanochat miniseries of models.
- To add new abilities to nanochat, see [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164).
- To customize your nanochat, see [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139) in Discussions, which describes how you can tune your nanochat's personality through synthetic data generation and mixing that data into the SFT stage.
- [Oct 13 2025: original nanochat post](https://github.com/karpathy/nanochat/discussions/1) introducing nanochat, though now it contains some deprecated information and the model is a lot older (with worse results) than current master.

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # Example synthetic data for identity
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # Pretraining data shard generation
├── nanochat
│   ├── __init__.py                 # empty
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── optim.py                    # AdamW + Muon optimizer, 1GPU and distributed
│   ├── report.py                   # Utilities for writing the nanochat Report
│   ├── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
│   └── ui.html                     # HTML/CSS/JS for nanochat frontend
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Miniseries training script
│   ├── runcpu.sh                   # Small example of how to run on CPU/MPS
│   ├── scaling_laws.sh             # Scaling laws experiments
│   └── speedrun.sh                 # Train the ~$100 nanochat d20
├── scripts
│   ├── base_eval.py                # Base model: CORE score, bits per byte, samples
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model: talk to over CLI
│   ├── chat_eval.py                # Chat model: eval tasks
│   ├── chat_rl.py                  # Chat model: reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── chat_web.py                 # Chat model: talk to over WebUI
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # Make Task from arbitrary jsonl convos
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   ├── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
│   └── spellingbee.py              # Task teaching model to spell/count letters
├── tests
│   └── test_engine.py
└── uv.lock
```

## Contributing

The goal of nanochat is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there are no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a ChatGPT model you can talk to. Currently, the most interesting part personally is speeding up the latency to GPT-2 (i.e. getting a CORE score above 0.256525). Currently this takes ~3 hours, but by improving the pretraining stage we can improve this further.

Current AI policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT

### Third-party licenses

[nanochat/mamba3.py](nanochat/mamba3.py) is derived from [mamba3-minimal](https://github.com/VikramLex/mamba3-minimal) (Copyright 2026 Vikram Karlex) and is licensed under the **Apache License 2.0**. See [nanochat/NOTICE](nanochat/NOTICE) for the full attribution and a summary of modifications.
