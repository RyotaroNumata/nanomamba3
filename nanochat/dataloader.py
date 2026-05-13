"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import os
import random
import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files, list_parquet_files_bilingual


def _single_lang_iter(parquet_paths, ddp_rank, ddp_world_size, tokenizer_batch_size,
                      resume_pq_idx=0, resume_rg_idx=None):
    """
    Infinite iterator over (text_batch, pq_idx, rg_idx) for a single language.

    Handles DDP sharding (each rank processes different row groups within a file)
    and approximate resume (skips ahead to the last checkpoint position).
    """
    first_pass = True
    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            if first_pass and resume_rg_idx is not None and pq_idx == resume_pq_idx:
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], pq_idx, rg_idx
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False


def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, state_dict)
    where text_batch is a list of document strings and state_dict tracks position for
    resumption.

    When NANOCHAT_JA_RATIO > 0 and split == "train":
        Uses document-level random mixing: each refill independently draws from EN or JA
        with probability (1-ja_ratio) or ja_ratio. This avoids periodic loss spikes that
        occur with shard-level interleaving.

    When NANOCHAT_JA_RATIO > 0 and split == "val":
        Uses shard-level interleaving of the last EN and JA shards (unchanged behavior).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    ja_ratio = float(os.environ.get("NANOCHAT_JA_RATIO", "0.0"))
    warn_on_legacy = ddp_rank == 0 and split == "train"

    if ja_ratio > 0.0 and split == "train":
        # Document-level bilingual mixing for train split
        en_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)[:-1]  # exclude last shard (val)
        ja_paths = list_parquet_files(lang="ja")[:-1]  # exclude last shard (val)

        assert len(en_paths) > 0, "No EN dataset parquet files found, did you run dataset.py?"
        assert len(ja_paths) > 0, (
            "No JA dataset parquet files found. "
            "Set NANOCHAT_JA_RATIO=0 or run: python -m nanochat.dataset -n 4 -l ja"
        )

        if ddp_rank == 0:
            print(f"Bilingual dataloader: {len(en_paths)} EN + {len(ja_paths)} JA shards "
                  f"(NANOCHAT_JA_RATIO={ja_ratio}, effective ratio={ja_ratio:.2f})")

        # Resume state — support both old format (pq_idx) and new format (en_pq_idx)
        if resume_state_dict is not None:
            en_resume_pq = resume_state_dict.get("en_pq_idx", resume_state_dict.get("pq_idx", 0))
            en_resume_rg = resume_state_dict.get("en_rg_idx", resume_state_dict.get("rg_idx", None))
            ja_resume_pq = resume_state_dict.get("ja_pq_idx", 0)
            ja_resume_rg = resume_state_dict.get("ja_rg_idx", None)
            epoch = resume_state_dict.get("epoch", 1)
        else:
            en_resume_pq, en_resume_rg = 0, None
            ja_resume_pq, ja_resume_rg = 0, None
            epoch = 1

        en_iter = _single_lang_iter(en_paths, ddp_rank, ddp_world_size, tokenizer_batch_size,
                                    en_resume_pq, en_resume_rg)
        ja_iter = _single_lang_iter(ja_paths, ddp_rank, ddp_world_size, tokenizer_batch_size,
                                    ja_resume_pq, ja_resume_rg)

        # Track last-seen positions for checkpoint state
        en_pq_idx = en_resume_pq
        en_rg_idx = en_resume_rg if en_resume_rg is not None else ddp_rank
        ja_pq_idx = ja_resume_pq
        ja_rg_idx = ja_resume_rg if ja_resume_rg is not None else ddp_rank

        while True:
            if random.random() < ja_ratio:
                batch, ja_pq_idx, ja_rg_idx = next(ja_iter)
            else:
                batch, en_pq_idx, en_rg_idx = next(en_iter)

            yield batch, {
                "pq_idx": en_pq_idx,    # backward compat: maps to EN position for display
                "rg_idx": en_rg_idx,    # backward compat: maps to EN position for display
                "epoch": epoch,
                "en_pq_idx": en_pq_idx,
                "en_rg_idx": en_rg_idx,
                "ja_pq_idx": ja_pq_idx,
                "ja_rg_idx": ja_rg_idx,
            }

    else:
        # Monolingual EN-only, or val split (uses shard-level interleave for bilingual val)
        if ja_ratio > 0.0:
            # val split: interleave last EN shard + last JA shard
            parquet_paths = list_parquet_files_bilingual(split, ja_ratio, warn_on_legacy=warn_on_legacy)
        else:
            parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
            parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
            if ddp_rank == 0 and split == "train":
                print(f"Monolingual dataloader (EN only): {len(parquet_paths)} shards"
                      f" — set NANOCHAT_JA_RATIO>0 to enable bilingual mixing")

        assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"

        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
        first_pass = True
        pq_idx = resume_pq_idx
        epoch = resume_epoch

        while True:  # iterate infinitely (multi-epoch)
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths):
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # Start from resume point if resuming on same file, otherwise from DDP rank
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1  # advance by 1 so we don't repeat data after resuming
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None  # only do this once
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist()
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
                    rg_idx += ddp_world_size
                pq_idx += 1
            first_pass = False
            epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    _loader_state = {"pq_idx": 0, "rg_idx": 0, "epoch": 1}

    def refill_buffer():
        nonlocal _loader_state
        doc_batch, _loader_state = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, _loader_state

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
