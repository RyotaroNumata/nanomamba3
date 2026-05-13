"""
Japanese pretraining evaluation script.

Evaluates a base model's Japanese language capability via:
  --eval ja_bpb     : Bits per byte on Japanese Wikipedia
  --eval ja_bench   : JCommonsenseQA accuracy (5-choice, loss-based)
  --eval ja_sample  : Generate completions from Japanese prompts

Default: all three.

Examples:
    # Evaluate a nanochat model
    python -m scripts.ja_eval --model-tag d24

    # Evaluate a HuggingFace model
    python -m scripts.ja_eval --hf-path openai-community/gpt2 --eval ja_bpb,ja_bench

    # Quick run (fewer tokens / examples)
    python -m scripts.ja_eval --model-tag d24 --max-tokens 1000000 --max-examples 200
"""

import random
import argparse

import torch

from nanochat.common import (
    compute_init, compute_cleanup, print0,
    autodetect_device_type,
)
from nanochat.tokenizer import get_token_bytes, HuggingFaceTokenizer
from nanochat.checkpoint_manager import load_model
from nanochat.loss_eval import evaluate_bpb
from nanochat.core_eval import evaluate_task
from nanochat.engine import Engine
from nanochat.report import get_report

# ---------------------------------------------------------------------------
# HuggingFace model wrapper (mirrors base_eval.py)

class ModelWrapper:
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )

    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path, device):
    from transformers import AutoModelForCausalLM
    print0(f"Loading HuggingFace model from: {hf_path}")
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "gpt2" in hf_path else None
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return ModelWrapper(model, max_seq_len=max_seq_len), tokenizer


def get_hf_token_bytes(tokenizer, device="cpu"):
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode("utf-8"))
    return token_bytes


# ---------------------------------------------------------------------------
# ja_bpb: Bits per byte on Japanese Wikipedia

def _ja_wikipedia_batches(tokenizer, batch_size, seq_len, max_tokens, device):
    """
    Stream Japanese Wikipedia, tokenize in mini-batches, pack into (x, y) tensors.
    Stops after yielding at least max_tokens tokens worth of batches.
    """
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
    bos = tokenizer.get_bos_token_id()
    tokens_per_batch = batch_size * seq_len

    text_batch = []
    token_buffer = []
    total_yielded = 0

    for article in ds:
        text_batch.append(article["text"])
        if len(text_batch) >= 16:
            token_lists = tokenizer(text_batch, prepend=bos)
            for toks in token_lists:
                token_buffer.extend(toks)
            text_batch = []

            while len(token_buffer) >= tokens_per_batch + 1:
                chunk = token_buffer[:tokens_per_batch + 1]
                token_buffer = token_buffer[tokens_per_batch:]
                x = torch.tensor(chunk[:tokens_per_batch], dtype=torch.long, device=device).view(batch_size, seq_len)
                y = torch.tensor(chunk[1:tokens_per_batch + 1], dtype=torch.long, device=device).view(batch_size, seq_len)
                yield x, y
                total_yielded += tokens_per_batch
                if total_yielded >= max_tokens:
                    return


def evaluate_ja_bpb(model, tokenizer, device, token_bytes, batch_size, seq_len, max_tokens):
    """Compute BPB on streaming Japanese Wikipedia."""
    steps = max(1, max_tokens // (batch_size * seq_len))
    print0(f"ja_bpb: batch_size={batch_size}, seq_len={seq_len}, steps={steps} (~{steps * batch_size * seq_len:,} tokens)")
    batches = _ja_wikipedia_batches(tokenizer, batch_size, seq_len, max_tokens, device)
    return evaluate_bpb(model, batches, steps, token_bytes)


# ---------------------------------------------------------------------------
# ja_bench: JCommonsenseQA accuracy (loss-based, no SFT required)

def _load_jcommonsenseqa(max_examples):
    """
    Load JCommonsenseQA validation split and convert to core_eval multiple_choice format.

    Each item:
        query   : the Japanese question text
        choices : list of 5 Japanese answer strings
        gold    : index (0-4) of the correct answer
    """
    from datasets import load_dataset

    ds = load_dataset("sbintuitions/JCommonsenseQA", split="validation")
    data = [
        {
            "query": row["question"],
            "choices": [row[f"choice{i}"] for i in range(5)],
            "gold": row["label"],
        }
        for row in ds
    ]
    rng = random.Random(42)
    rng.shuffle(data)
    if max_examples > 0:
        data = data[:max_examples]
    return data


def evaluate_ja_bench(model, tokenizer, device, max_examples):
    """Evaluate JCommonsenseQA accuracy using loss-based multiple-choice scoring."""
    print0("Loading JCommonsenseQA (validation)...")
    data = _load_jcommonsenseqa(max_examples)
    print0(f"Evaluating {len(data)} examples...")

    task_meta = {
        "task_type": "multiple_choice",
        "num_fewshot": 0,
        "continuation_delimiter": "\n",
    }
    accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
    # 5-choice random baseline is 20%
    centered = (accuracy - 0.20) / (1.0 - 0.20)
    print0(f"JCommonsenseQA  accuracy: {accuracy:.4f} | centered (vs random=0.20): {centered:.4f}")
    return accuracy, centered


# ---------------------------------------------------------------------------
# ja_sample: qualitative generation from Japanese prompts

JA_PROMPTS = [
    "東京は日本の",
    "水の化学式は",
    "富士山の高さは",
    "日本で一番長い川は",
    "桜の花が咲く季節は",
    "人工知能とは",
    "もし明日が休日なら、",
]


def evaluate_ja_sample(model, tokenizer, device):
    """Generate short completions for a fixed set of Japanese prompts."""
    engine = Engine(model, tokenizer)
    samples = []
    print0("\nJapanese conditioned samples (greedy, max 32 tokens):")
    for prompt in JA_PROMPTS:
        tokens = tokenizer(prompt, prepend="<|bos|>")
        sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=32, temperature=0)
        generated = tokenizer.decode(sample[0][len(tokens):])
        print0("-" * 60)
        print0(f"Q: {prompt}")
        print0(f"A: {generated}")
        samples.append(f"Q: {prompt}\nA: {generated}")
    return samples


# ---------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Japanese pretraining evaluation")
    parser.add_argument(
        "--eval", type=str, default="ja_bpb,ja_bench,ja_sample",
        help="Comma-separated eval modes: ja_bpb,ja_bench,ja_sample (default: all)",
    )
    parser.add_argument("--hf-path", type=str, default=None, help="HuggingFace model path")
    parser.add_argument("--model-tag", type=str, default=None, help="nanochat model tag")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to load (default: last)")
    parser.add_argument(
        "--max-examples", type=int, default=-1,
        help="Max examples for ja_bench (-1 = all 1,119)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=5_000_000,
        help="Total tokens to evaluate for ja_bpb (default: 5M)",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for ja_bpb")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length for ja_bpb (default: model's own)")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    args = parser.parse_args()

    eval_modes = {m.strip() for m in args.eval.split(",")}
    valid_modes = {"ja_bpb", "ja_bench", "ja_sample"}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    # Single-GPU: compute_init with no distributed setup
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    is_hf = args.hf_path is not None
    if is_hf:
        model, tokenizer = load_hf_model(args.hf_path, device)
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_name = args.hf_path
        seq_len = args.seq_len or model.max_seq_len or 1024
    else:
        model, tokenizer, meta = load_model(
            "base", device, phase="eval", model_tag=args.model_tag, step=args.step
        )
        token_bytes = get_token_bytes(device=device)
        model_name = f"base_model (step {meta['step']})"
        seq_len = args.seq_len or meta["model_config"]["sequence_len"]

    print0(f"Model   : {model_name}")
    print0(f"Seq len : {seq_len}")
    print0(f"Eval    : {', '.join(sorted(eval_modes))}")

    results = {"model": model_name}
    samples = []

    # --- ja_bpb ---
    if "ja_bpb" in eval_modes:
        print0("\n" + "=" * 70)
        print0("Japanese BPB  (Wikipedia-ja)")
        print0("=" * 70)
        bpb = evaluate_ja_bpb(
            model, tokenizer, device, token_bytes,
            args.batch_size, seq_len, args.max_tokens,
        )
        results["ja_bpb"] = bpb
        print0(f"Result  ja_bpb: {bpb:.6f}")

    # --- ja_bench ---
    if "ja_bench" in eval_modes:
        print0("\n" + "=" * 70)
        print0("JCommonsenseQA")
        print0("=" * 70)
        accuracy, centered = evaluate_ja_bench(model, tokenizer, device, args.max_examples)
        results["jcommonsenseqa_accuracy"] = accuracy
        results["jcommonsenseqa_centered"] = centered

    # --- ja_sample ---
    if "ja_sample" in eval_modes:
        print0("\n" + "=" * 70)
        print0("Japanese Samples")
        print0("=" * 70)
        if is_hf:
            print0("Skipping ja_sample: Engine not supported for HuggingFace models")
        else:
            samples = evaluate_ja_sample(model, tokenizer, device)

    # --- Report ---
    report_data = [results]
    if samples:
        report_data.append({f"ja_sample_{i}": s for i, s in enumerate(samples)})
    get_report().log(section="Japanese evaluation", data=report_data)
    print0("\nDone. Results logged to report.")

    compute_cleanup()


if __name__ == "__main__":
    main()
