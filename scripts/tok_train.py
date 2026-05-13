"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
import os
import time
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

# When NANOCHAT_JA_RATIO > 0, lower the default max_chars to 500M to avoid
# OOM during Japanese BPE training (2B chars causes memory exhaustion).
_ja_ratio = float(os.environ.get("NANOCHAT_JA_RATIO", "0.0"))
_default_max_chars = 500_000_000 if _ja_ratio > 0.0 else 2_000_000_000

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=_default_max_chars,
                    help='Maximum characters to train on (default: 500M when JA enabled, 2B otherwise)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")
print(f"NANOCHAT_JA_RATIO: {_ja_ratio}")

# -----------------------------------------------------------------------------
# Text iterator

def _iter_docs(lang, budget):
    """Yield documents from one language up to `budget` characters."""
    for batch in parquets_iter_batched(split="train", lang=lang):
        for doc in batch:
            text = doc[:args.doc_cap] if len(doc) > args.doc_cap else doc
            budget -= len(text)
            yield text
            if budget <= 0:
                return

def text_iterator():
    """
    Yield documents for tokenizer training.

    When NANOCHAT_JA_RATIO > 0, mixes EN and JA documents according to the
    ratio so the BPE vocabulary covers both languages.
    The total character budget is split proportionally between EN and JA.
    """
    if _ja_ratio <= 0.0:
        # English-only (original behaviour)
        nchars = 0
        for batch in parquets_iter_batched(split="train"):
            for doc in batch:
                text = doc[:args.doc_cap] if len(doc) > args.doc_cap else doc
                nchars += len(text)
                yield text
                if nchars >= args.max_chars:
                    return
    else:
        # Bilingual: split the char budget proportionally, then interleave
        ja_budget = int(args.max_chars * _ja_ratio)
        en_budget = args.max_chars - ja_budget
        print(f"Tokenizer training: EN budget {en_budget:,} chars, JA budget {ja_budget:,} chars")

        en_iter = _iter_docs("en", en_budget)
        ja_iter = _iter_docs("ja", ja_budget)

        # Interleave EN and JA in round-robin at the shard level
        en_done, ja_done = False, False
        en_count, ja_count = 0, 0
        # Desired ratio: yield 1 JA for every (1/ja_ratio - 1) EN docs
        ja_per_en = _ja_ratio / (1.0 - _ja_ratio)  # e.g. 0.25 for ja_ratio=0.2
        for doc in en_iter:
            yield doc
            en_count += 1
            # Insert JA docs to maintain the ratio
            while en_count * ja_per_en > ja_count:
                try:
                    yield next(ja_iter)
                    ja_count += 1
                except StopIteration:
                    break

text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])
