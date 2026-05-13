"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Language-specific data configuration

@dataclass
class DataConfig:
    """Configuration for a language-specific pretraining data source."""
    base_url: str
    max_shard: int          # index of the last shard (inclusive)
    data_dir: str           # local directory to store parquet files
    index_to_filename: object  # callable: index -> filename string
    text_column: str = "text"


base_dir = get_base_dir()

DATA_CONFIGS = {
    "en": DataConfig(
        base_url="https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main",
        max_shard=6542,
        data_dir=os.path.join(base_dir, "base_data_climbmix"),
        index_to_filename=lambda i: f"shard_{i:05d}.parquet",
    ),
    "ja": DataConfig(
        base_url="https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese/resolve/main/data",
        max_shard=1238,  # files: train-00000-of-01239.parquet ... train-01238-of-01239.parquet
        data_dir=os.path.join(base_dir, "base_data_ja"),
        index_to_filename=lambda i: f"train-{i:05d}-of-01239.parquet",
    ),
}

# Legacy aliases (used by other modules that import these directly)
BASE_URL = DATA_CONFIGS["en"].base_url
MAX_SHARD = DATA_CONFIGS["en"].max_shard
index_to_filename = DATA_CONFIGS["en"].index_to_filename
DATA_DIR = DATA_CONFIGS["en"].data_dir

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None, warn_on_legacy=False, lang=None):
    """
    Returns full paths to all parquet files in the data directory.

    Priority: explicit data_dir > lang-specific dir > default EN dir.
    When warn_on_legacy=True and the EN ClimbMix dir is missing, prints an
    upgrade notice and falls back to the old FinewebEdu directory.
    """
    if data_dir is None:
        if lang is not None and lang != "en":
            data_dir = DATA_CONFIGS[lang].data_dir
        else:
            # Default: English (with legacy fallback)
            en_dir = DATA_CONFIGS["en"].data_dir
            if not os.path.exists(en_dir):
                if warn_on_legacy:
                    print()
                    print("=" * 80)
                    print("  WARNING: DATASET UPGRADE REQUIRED")
                    print("=" * 80)
                    print()
                    print(f"  Could not find: {en_dir}")
                    print()
                    print("  nanochat recently switched from FinewebEdu-100B to ClimbMix-400B.")
                    print("  Everyone who does `git pull` as of March 4, 2026 is expected to see this message.")
                    print("  To upgrade to the new ClimbMix-400B dataset, run these two commands:")
                    print()
                    print("    python -m nanochat.dataset -n 170     # download ~170 shards")
                    print("    python -m scripts.tok_train           # re-train tokenizer on new ClimbMix data")
                    print()
                    print("  For now, falling back to your old FinewebEdu-100B dataset...")
                    print("=" * 80)
                    print()
                data_dir = os.path.join(base_dir, "base_data")
            else:
                data_dir = en_dir

    if not os.path.exists(data_dir):
        return []
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    return [os.path.join(data_dir, f) for f in parquet_files]


def list_parquet_files_bilingual(split, ja_ratio, warn_on_legacy=False):
    """
    Returns an interleaved list of EN and JA parquet paths for the given split.

    JA files are cycled (repeated) if there are fewer than the ratio demands.
    For example, with ja_ratio=0.2 and 100 EN files, ~25 JA slots are needed
    (100 * 0.2 / 0.8 ≈ 25).  If only 10 JA files exist they cycle 2-3 times.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    en_all = list_parquet_files(warn_on_legacy=warn_on_legacy)
    ja_all = list_parquet_files(lang="ja")

    en_paths = en_all[:-1] if split == "train" else en_all[-1:]
    ja_paths = ja_all[:-1] if split == "train" else ja_all[-1:]

    if not ja_paths:
        return en_paths

    # How many JA slots: en_count * r / (1 - r)
    en_count = len(en_paths)
    ja_target = max(1, round(en_count * ja_ratio / (1.0 - ja_ratio)))

    # Cycle JA paths to fill ja_target slots
    ja_cycled = [ja_paths[i % len(ja_paths)] for i in range(ja_target)]

    # Interleave: insert one JA path every `stride` EN paths
    stride = max(1, en_count // ja_target)
    result = []
    ja_idx = 0
    for i, en_path in enumerate(en_paths):
        result.append(en_path)
        if ja_idx < len(ja_cycled) and (i + 1) % stride == 0:
            result.append(ja_cycled[ja_idx])
            ja_idx += 1
    result.extend(ja_cycled[ja_idx:])  # any leftover JA slots at the end

    return result


def parquets_iter_batched(split, start=0, step=1, lang=None):
    """
    Iterate through the dataset in batches of underlying row_groups.

    - split: "train" or "val" (last file is always val)
    - start/step: for DDP sharding (start=rank, step=world_size)
    - lang: "en" or "ja".  Defaults to "en".
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    cfg = DATA_CONFIGS.get(lang or "en", DATA_CONFIGS["en"])
    parquet_paths = list_parquet_files(lang=lang)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column(cfg.text_column).to_pylist()
            yield texts


# -----------------------------------------------------------------------------
def download_single_file(index, lang="en"):
    """Downloads a single parquet shard for the given language, with retry."""
    cfg = DATA_CONFIGS[lang]
    data_dir = cfg.data_dir
    os.makedirs(data_dir, exist_ok=True)

    filename = cfg.index_to_filename(index)
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    url = f"{cfg.base_url}/{filename}"
    print(f"Downloading {filename}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download (-1 = all)")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers")
    parser.add_argument("-l", "--lang", type=str, default="en", choices=list(DATA_CONFIGS.keys()),
                        help="Language of the dataset to download (default: en)")
    args = parser.parse_args()

    cfg = DATA_CONFIGS[args.lang]
    os.makedirs(cfg.data_dir, exist_ok=True)

    num_train_shards = cfg.max_shard if args.num_files == -1 else min(args.num_files, cfg.max_shard)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(cfg.max_shard)  # always include the validation shard

    print(f"Language: {args.lang}")
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {cfg.data_dir}")
    print()

    download_fn = partial(download_single_file, lang=args.lang)
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_fn, ids_to_download)

    successful = sum(1 for ok in results if ok)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {cfg.data_dir}")
