"""
Inference speed benchmark: Mamba-3 MIMO vs GPT (bilingual_v2)
Covers: CUDA (PyTorch), CPU (PyTorch), ONNX fp32 (CPU), ONNX int8 (CPU)

Usage:
    python inference_bench.py                    # auto-detect GPU/CPU
    python inference_bench.py --device cpu       # CPU only
    python inference_bench.py --no-onnx          # skip ONNX benchmarks
"""
import argparse
import time
import torch
from nanochat.common import compute_init
from nanochat.checkpoint_manager import load_model

PROMPT_LENS  = [32, 128, 512]
GEN_TOKENS   = 100
WARMUP_STEPS = 5
REPEAT       = 3

# ---------------------------------------------------------------------------

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def bench_pytorch(model_source, model_tag, label, device):
    model, tokenizer, meta = load_model(
        model_source, device, phase="eval", model_tag=model_tag
    )
    arch = meta.get("model_config", {}).get("model_arch", "transformer")

    print(f"\n{'='*62}")
    print(f"  {label}  [{arch}]  device={device}")
    print(f"{'='*62}")
    print(f"{'prompt':>8} | {'tok/s':>8} | {'1st-tok ms':>11} | {'total ms':>9}")
    print(f"{'-'*8}-+-{'-'*8}-+-{'-'*11}-+-{'-'*9}")

    bos = tokenizer.get_bos_token_id()
    for plen in PROMPT_LENS:
        tokens = [bos] + [42] * (plen - 1)

        # warmup
        for _ in range(WARMUP_STEPS):
            with torch.inference_mode():
                list(model.generate(tokens, max_tokens=WARMUP_STEPS, temperature=0.0))

        times, first_toks = [], []
        for _ in range(REPEAT):
            sync(device)
            t0 = time.perf_counter()
            first_tok = None
            with torch.inference_mode():
                for tok in model.generate(tokens, max_tokens=GEN_TOKENS, temperature=0.0):
                    if first_tok is None:
                        sync(device)
                        first_tok = time.perf_counter() - t0
            sync(device)
            total = time.perf_counter() - t0
            times.append(total)
            first_toks.append(first_tok)

        tps    = GEN_TOKENS / (sum(times) / REPEAT)
        first  = sum(first_toks) / REPEAT * 1000
        total  = sum(times) / REPEAT * 1000
        print(f"{plen:>8} | {tps:>8.1f} | {first:>11.1f} | {total:>9.1f}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def bench_onnx_with_prefill(decode_path, prefill_path, label):
    """Benchmark ONNX with chunk-prefill vs token-by-token, showing both."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print(f"\n[skip] onnxruntime not installed — skipping {label}")
        return

    import os
    if not os.path.exists(decode_path) or not os.path.exists(prefill_path):
        print(f"\n[skip] ONNX files not found — skipping {label}")
        return

    from nanochat.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(decode_path, sess_opts, providers=["CPUExecutionProvider"])
    sess_p = ort.InferenceSession(prefill_path, sess_opts, providers=["CPUExecutionProvider"])

    n_layer = sum(1 for inp in sess.get_inputs() if inp.name.startswith("ssm_"))
    meta = {inp.name: inp.shape for inp in sess.get_inputs()}
    chunk_size = next(inp.shape[0] for inp in sess_p.get_inputs() if inp.name == "token_ids")

    def make_states():
        states = {}
        for prefix in ("ssm", "k", "v", "angle"):
            for i in range(n_layer):
                name = f"{prefix}_{i}"
                states[name] = np.zeros(meta[name], dtype=np.float32)
        return states

    def step(token_id, states):
        feed = {"token_id": np.array([token_id], dtype=np.int64)}
        feed.update(states)
        outputs = sess.run(None, feed)
        names = [o.name for o in sess.get_outputs()][1:]
        new_states = {n.removeprefix("new_"): outputs[i+1] for i, n in enumerate(names)}
        return outputs[0], new_states

    def chunk_step(token_ids, states):
        feed = {"token_ids": np.array(token_ids, dtype=np.int64)}
        feed.update(states)
        outputs = sess_p.run(None, feed)
        names = [o.name for o in sess_p.get_outputs()][1:]
        new_states = {n.removeprefix("new_"): outputs[i+1] for i, n in enumerate(names)}
        return outputs[0], new_states

    print(f"\n{'='*72}")
    print(f"  {label}  [ONNX / CPU, chunk_size={chunk_size}]")
    print(f"{'='*72}")
    print(f"{'prompt':>8} | {'no-prefill tok/s':>16} | {'chunk-prefill tok/s':>18} | {'speedup':>7}")
    print(f"{'-'*8}-+-{'-'*16}-+-{'-'*18}-+-{'-'*7}")

    bos = tokenizer.get_bos_token_id()
    for plen in PROMPT_LENS:
        tokens = [bos] + [42] * (plen - 1)

        # — token-by-token —
        times_tt = []
        for _ in range(REPEAT):
            st = make_states()
            t0 = time.perf_counter()
            for tok in tokens:
                logits, st = step(tok, st)
            next_id = int(np.argmax(logits))
            for _ in range(GEN_TOKENS):
                logits, st = step(next_id, st)
                next_id = int(np.argmax(logits))
            times_tt.append(time.perf_counter() - t0)
        tps_tt = GEN_TOKENS / (sum(times_tt) / REPEAT)

        # — chunk prefill —
        times_cp = []
        for _ in range(REPEAT):
            st = make_states()
            t0 = time.perf_counter()
            n_chunked = (len(tokens) // chunk_size) * chunk_size
            for i in range(0, n_chunked, chunk_size):
                logits, st = chunk_step(tokens[i:i + chunk_size], st)
            for tok in tokens[n_chunked:]:
                logits, st = step(tok, st)
            next_id = int(np.argmax(logits))
            for _ in range(GEN_TOKENS):
                logits, st = step(next_id, st)
                next_id = int(np.argmax(logits))
            times_cp.append(time.perf_counter() - t0)
        tps_cp = GEN_TOKENS / (sum(times_cp) / REPEAT)

        speedup = tps_cp / tps_tt
        print(f"{plen:>8} | {tps_tt:>16.1f} | {tps_cp:>18.1f} | {speedup:>6.2f}x")


def bench_onnx(onnx_path, label):
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print(f"\n[skip] onnxruntime not installed — skipping {label}")
        return

    import os
    if not os.path.exists(onnx_path):
        print(f"\n[skip] {onnx_path} not found — skipping {label}")
        return

    from nanochat.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(onnx_path, sess_opts,
                                providers=["CPUExecutionProvider"])

    n_layer = sum(1 for inp in sess.get_inputs() if inp.name.startswith("ssm_"))
    meta    = {inp.name: inp.shape for inp in sess.get_inputs()}

    def make_states():
        states = {}
        for prefix in ("ssm", "k", "v", "angle"):
            for i in range(n_layer):
                name = f"{prefix}_{i}"
                states[name] = np.zeros(meta[name], dtype=np.float32)
        return states

    def step(token_id, states):
        feed = {"token_id": np.array([token_id], dtype=np.int64)}
        feed.update(states)
        outputs = sess.run(None, feed)
        names = [o.name for o in sess.get_outputs()][1:]
        new_states = {n.removeprefix("new_"): outputs[i+1] for i, n in enumerate(names)}
        return outputs[0], new_states

    print(f"\n{'='*62}")
    print(f"  {label}  [ONNX / CPU]")
    print(f"{'='*62}")
    print(f"{'prompt':>8} | {'tok/s':>8} | {'1st-tok ms':>11} | {'total ms':>9}")
    print(f"{'-'*8}-+-{'-'*8}-+-{'-'*11}-+-{'-'*9}")

    bos = tokenizer.get_bos_token_id()
    for plen in PROMPT_LENS:
        tokens = [bos] + [42] * (plen - 1)

        # warmup
        for _ in range(WARMUP_STEPS):
            st = make_states()
            for tok in tokens:
                logits, st = step(tok, st)
            for _ in range(WARMUP_STEPS):
                logits, st = step(int(np.argmax(logits)), st)

        times, first_toks = [], []
        for _ in range(REPEAT):
            st = make_states()
            t0 = time.perf_counter()
            # prefill
            for tok in tokens:
                logits, st = step(tok, st)
            first_tok = time.perf_counter() - t0
            # decode
            next_id = int(np.argmax(logits))
            for _ in range(GEN_TOKENS):
                logits, st = step(next_id, st)
                next_id = int(np.argmax(logits))
            total = time.perf_counter() - t0
            times.append(total)
            first_toks.append(first_tok)

        tps   = GEN_TOKENS / (sum(times) / REPEAT)
        first = sum(first_toks) / REPEAT * 1000
        total = sum(times) / REPEAT * 1000
        print(f"{plen:>8} | {tps:>8.1f} | {first:>11.1f} | {total:>9.1f}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for PyTorch benchmarks")
    parser.add_argument("--no-onnx", action="store_true",
                        help="Skip ONNX benchmarks")
    parser.add_argument("--onnx-fp32", default="/tmp/mamba3_step_fp32.onnx",
                        help="Path to ONNX fp32 model")
    parser.add_argument("--onnx-int8", default="/tmp/mamba3_step_int8.onnx",
                        help="Path to ONNX int8 model")
    parser.add_argument("--onnx-prefill", default="/tmp/mamba3_step_fp32_prefill.onnx",
                        help="Path to ONNX fp32 chunk-prefill model")
    parser.add_argument("--onnx-int8-prefill", default="/tmp/mamba3_step_int8_prefill.onnx",
                        help="Path to ONNX int8 chunk-prefill model")
    parser.add_argument("--no-pytorch", action="store_true",
                        help="Skip PyTorch benchmarks")
    args = parser.parse_args()

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Prompt lengths: {PROMPT_LENS}  |  Generate: {GEN_TOKENS} tokens  |  Repeat: {REPEAT}x")

    # ── PyTorch benchmarks ──────────────────────────────────────────────────
    if not args.no_pytorch:
        bench_pytorch("sft", "mamba3_mimo_r2_10k_sft",
                      "Mamba-3 MIMO rank-2 (SFT)", device)
        bench_pytorch("sft", "bilingual_v2",
                      "GPT Transformer bilingual_v2 (SFT)", device)

        # ── CPU PyTorch (only if GPU was the primary device) ───────────────
        if device.type == "cuda":
            cpu = torch.device("cpu")
            bench_pytorch("sft", "mamba3_mimo_r2_10k_sft",
                          "Mamba-3 MIMO rank-2 (SFT)", cpu)

    # ── ONNX benchmarks (CPU) ───────────────────────────────────────────────
    if not args.no_onnx:
        bench_onnx_with_prefill(args.onnx_fp32, args.onnx_prefill,
                                "Mamba-3 ONNX fp32: token-by-token vs chunk-prefill")
        bench_onnx_with_prefill(args.onnx_int8, args.onnx_int8_prefill,
                                "Mamba-3 ONNX int8: token-by-token vs chunk-prefill")
        bench_onnx(args.onnx_fp32, "Mamba-3 ONNX fp32")
        bench_onnx(args.onnx_int8, "Mamba-3 ONNX int8")


if __name__ == "__main__":
    main()
