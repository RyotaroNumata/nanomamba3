"""
Export Mamba3 decode-step to ONNX.

The exported graph takes a single token + all layer SSM states and returns
next-token logits + updated states. This is the minimal unit needed for
autoregressive generation on mobile / ONNX Runtime.

Usage:
    python -m scripts.export_onnx --model-tag mamba3_mimo_r2_10k_sft --output /tmp/mamba3_step.onnx
    python -m scripts.export_onnx --model-tag mamba3_mimo_r2_10k_sft --output /tmp/mamba3_step.onnx --verify
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import autodetect_device_type, compute_init
from nanochat.checkpoint_manager import load_model
from nanochat.mamba3 import InferenceCache


# ---------------------------------------------------------------------------
# Wrapper: flattens all layer states into individual tensors for ONNX
# ---------------------------------------------------------------------------

class Mamba3DecodeStep(nn.Module):
    """
    Single decode-step wrapper for ONNX export.

    Inputs:
        token_id   : (1,)  int64
        ssm_*      : (1, nheads, headdim, d_state) float  — one per layer
        k_*        : (1, nheads, d_state [,R])     float  — one per layer
        v_*        : (1, nheads, headdim [,R])     float  — one per layer
        angle_*    : (1, nheads, num_rope_angles)  float  — one per layer

    Outputs:
        logits     : (vocab_size,)  float32
        new_ssm_*  : same shape as ssm_*  — one per layer
        new_k_*    : same shape as k_*    — one per layer
        new_v_*    : same shape as v_*    — one per layer
        new_angle_*: same shape as angle_ — one per layer
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, token_id, *flat_states):
        cfg = self.model.config
        n = cfg.n_layer
        # Reconstruct per-layer InferenceCache from flat tensors
        # Order: [ssm_0..ssm_n, k_0..k_n, v_0..v_n, angle_0..angle_n]
        ssm_states   = flat_states[0*n : 1*n]
        k_states     = flat_states[1*n : 2*n]
        v_states     = flat_states[2*n : 3*n]
        angle_states = flat_states[3*n : 4*n]

        caches = [
            InferenceCache(ssm_states[i], k_states[i], v_states[i], angle_states[i])
            for i in range(n)
        ]

        # Embedding
        x = self.model.transformer.wte(token_id.unsqueeze(0)).to(ssm_states[0].dtype)  # (1,1,d)

        # Run each layer
        new_caches = []
        for i, layer in enumerate(self.model.transformer.h):
            y, h_new = layer.mixer(layer.mixer_norm(x), caches[i])
            x = x + y
            x = x + layer.mlp(layer.mlp_norm(x))
            new_caches.append(h_new)

        # Final norm + lm_head
        # Use manual RMSNorm (F.rms_norm maps to aten::rms_norm, unsupported in ONNX opset<18)
        x = x[:, -1:]
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        logits = self.model.lm_head(x)[0, 0, :cfg.vocab_size].float()

        # Flatten output states
        out_ssm   = [c.ssm_state for c in new_caches]
        out_k     = [c.k_state   for c in new_caches]
        out_v     = [c.v_state   for c in new_caches]
        out_angle = [c.cum_angle for c in new_caches]

        return (logits, *out_ssm, *out_k, *out_v, *out_angle)


class Mamba3ChunkPrefill(nn.Module):
    """
    Chunk-prefill wrapper for ONNX export.

    Processes chunk_size tokens in one SSD forward pass, carrying over state
    from a previous chunk. Much faster than calling Mamba3DecodeStep chunk_size
    times: O(chunk_size²) SSD scan vs O(chunk_size) sequential decode calls.

    Inputs:
        token_ids  : (chunk_size,) int64  — one full chunk of prompt tokens
        ssm_*      : (1, nheads, headdim, d_state) float  — one per layer
        k_*        : (1, nheads, d_state [,R])     float  — one per layer
        v_*        : (1, nheads, headdim [,R])     float  — one per layer
        angle_*    : (1, nheads, num_rope_angles)  float  — one per layer

    Outputs:
        logits     : (vocab_size,)  float32  — logits for the last token in chunk
        new_ssm_*  : updated ssm state — one per layer
        new_k_*    : updated k state   — one per layer
        new_v_*    : updated v state   — one per layer
        new_angle_*: updated angle     — one per layer

    Usage: call repeatedly for each chunk_size-aligned segment of the prompt,
    then switch to Mamba3DecodeStep for the remaining tokens and generation.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, token_ids, *flat_states):
        cfg = self.model.config
        n = cfg.n_layer
        ssm_states   = flat_states[0*n : 1*n]
        k_states     = flat_states[1*n : 2*n]
        v_states     = flat_states[2*n : 3*n]
        angle_states = flat_states[3*n : 4*n]

        caches = [
            InferenceCache(ssm_states[i], k_states[i], v_states[i], angle_states[i])
            for i in range(n)
        ]

        # Embedding: token_ids is (Q,), add batch dim → (1, Q, d)
        x = self.model.transformer.wte(token_ids.unsqueeze(0)).to(ssm_states[0].dtype)

        # Run each layer with initial_state for cross-chunk continuity
        new_caches = []
        for i, layer in enumerate(self.model.transformer.h):
            y, h_new = layer.mixer(layer.mixer_norm(x), initial_state=caches[i])
            x = x + y
            x = x + layer.mlp(layer.mlp_norm(x))
            new_caches.append(h_new)

        # Final norm + lm_head on last token only
        # Manual RMSNorm (F.rms_norm → aten::rms_norm, unsupported in ONNX opset<18)
        x = x[:, -1:]
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        logits = self.model.lm_head(x)[0, 0, :cfg.vocab_size].float()

        out_ssm   = [c.ssm_state for c in new_caches]
        out_k     = [c.k_state   for c in new_caches]
        out_v     = [c.v_state   for c in new_caches]
        out_angle = [c.cum_angle for c in new_caches]

        return (logits, *out_ssm, *out_k, *out_v, *out_angle)


def make_dummy_states(model, device, dtype):
    """Create zero initial states for all layers."""
    cfg = model.config
    n = cfg.n_layer
    R = cfg.mimo_rank
    batch = 1

    ssm_list   = [torch.zeros(batch, cfg.nheads, cfg.headdim, cfg.d_state, device=device, dtype=dtype) for _ in range(n)]
    angle_list = [torch.zeros(batch, cfg.nheads, cfg.num_rope_angles,       device=device, dtype=dtype) for _ in range(n)]

    if R == 1:
        k_list = [torch.zeros(batch, cfg.nheads, cfg.d_state,              device=device, dtype=dtype) for _ in range(n)]
        v_list = [torch.zeros(batch, cfg.nheads, cfg.headdim,              device=device, dtype=dtype) for _ in range(n)]
    else:
        k_list = [torch.zeros(batch, cfg.nheads, cfg.d_state,  R,          device=device, dtype=dtype) for _ in range(n)]
        v_list = [torch.zeros(batch, cfg.nheads, cfg.headdim,  R,          device=device, dtype=dtype) for _ in range(n)]

    return ssm_list, k_list, v_list, angle_list


def build_io_names(n_layer, prefill=False):
    """Build ONNX input/output name lists.
    prefill=True: first input is token_ids (chunk_size,) instead of token_id (1,).
    """
    inputs  = ["token_ids"] if prefill else ["token_id"]
    outputs = ["logits"]
    for prefix in ("ssm", "k", "v", "angle"):
        for i in range(n_layer):
            inputs.append(f"{prefix}_{i}")
    for prefix in ("ssm", "k", "v", "angle"):
        for i in range(n_layer):
            outputs.append(f"new_{prefix}_{i}")
    return inputs, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--source", type=str, default="sft", choices=["base", "sft"])
    parser.add_argument("--output", type=str, default="/tmp/mamba3_step.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp32", action="store_true", help="Cast model to float32 before export (required for CPU/mobile)")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX output matches PyTorch")
    parser.add_argument("--export-prefill", action="store_true",
                        help="Also export chunk-prefill model to <output stem>_prefill.onnx")
    args = parser.parse_args()

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)

    print(f"Loading model (source={args.source}, tag={args.model_tag})...")
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag)
    model.eval()

    cfg = model.config
    dtype = next(model.parameters()).dtype
    n = cfg.n_layer
    R = cfg.mimo_rank
    print(f"Model: n_layer={n}, d_model={cfg.n_embd}, mimo_rank={R}, dtype={dtype}")

    if args.fp32:
        print("Casting model to float32...")
        model = model.float()
        dtype = torch.float32

    # Build wrapper
    wrapper = Mamba3DecodeStep(model).eval()

    # Dummy inputs
    token_id = torch.tensor([42], dtype=torch.long, device=device)
    ssm_list, k_list, v_list, angle_list = make_dummy_states(model, device, dtype)
    flat_states = (*ssm_list, *k_list, *v_list, *angle_list)

    input_names, output_names = build_io_names(n)

    # Dynamic axes: batch=1 fixed, but keep logits vocab dim static too
    dynamic_axes = {}  # all static for now (mobile-friendly)

    print(f"Exporting to {args.output} (opset={args.opset})...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (token_id, *flat_states),
            args.output,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,  # use legacy TorchScript-based exporter
        )
    print(f"Export done: {args.output}")

    if args.verify:
        print("\nVerifying ONNX output vs PyTorch...")
        import onnxruntime as ort
        import numpy as np

        # PyTorch reference
        with torch.no_grad():
            pt_out = wrapper(token_id, *flat_states)
        pt_logits = pt_out[0].cpu().float().numpy()

        # ONNX Runtime
        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        ort_inputs = {
            "token_id": token_id.cpu().numpy(),
        }
        for prefix, lst in [("ssm", ssm_list), ("k", k_list), ("v", v_list), ("angle", angle_list)]:
            for i, t in enumerate(lst):
                ort_inputs[f"{prefix}_{i}"] = t.cpu().float().numpy()
        ort_out = sess.run(None, ort_inputs)
        ort_logits = ort_out[0]

        max_diff = abs(pt_logits - ort_logits).max()
        top1_match = pt_logits.argmax() == ort_logits.argmax()
        print(f"  max_diff  : {max_diff:.6f}")
        print(f"  top1_match: {'✓' if top1_match else '✗'}")

        if max_diff < 0.01:
            print("  PASS: outputs match within tolerance")
        else:
            print("  WARN: large difference, check dtype handling")

    # Print decode-step model size
    import os
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nONNX decode-step file size: {size_mb:.1f} MB")

    # ── Chunk-prefill model export ──────────────────────────────────────────
    if args.export_prefill:
        prefill_path = args.output.replace(".onnx", "_prefill.onnx")
        prefill_wrapper = Mamba3ChunkPrefill(model).eval()

        # Dummy: one full chunk of tokens
        chunk_ids = torch.zeros(cfg.chunk_size, dtype=torch.long, device=device)
        prefill_input_names, prefill_output_names = build_io_names(n, prefill=True)

        print(f"\nExporting chunk-prefill model to {prefill_path} (chunk_size={cfg.chunk_size})...")
        with torch.no_grad():
            torch.onnx.export(
                prefill_wrapper,
                (chunk_ids, *flat_states),
                prefill_path,
                input_names=prefill_input_names,
                output_names=prefill_output_names,
                dynamic_axes={},
                opset_version=args.opset,
                do_constant_folding=True,
                dynamo=False,
            )
        print(f"Chunk-prefill export done: {prefill_path}")

        if args.verify:
            print("\nVerifying chunk-prefill ONNX output vs PyTorch...")
            import onnxruntime as ort
            import numpy as np

            with torch.no_grad():
                pt_out = prefill_wrapper(chunk_ids, *flat_states)
            pt_logits = pt_out[0].cpu().float().numpy()

            sess_p = ort.InferenceSession(prefill_path, providers=["CPUExecutionProvider"])
            ort_inputs_p = {"token_ids": chunk_ids.cpu().numpy()}
            for prefix, lst in [("ssm", ssm_list), ("k", k_list), ("v", v_list), ("angle", angle_list)]:
                for i, t in enumerate(lst):
                    ort_inputs_p[f"{prefix}_{i}"] = t.cpu().float().numpy()
            ort_out_p = sess_p.run(None, ort_inputs_p)
            ort_logits_p = ort_out_p[0]

            max_diff = abs(pt_logits - ort_logits_p).max()
            top1_match = pt_logits.argmax() == ort_logits_p.argmax()
            print(f"  max_diff  : {max_diff:.6f}")
            print(f"  top1_match: {'✓' if top1_match else '✗'}")
            print("  PASS" if max_diff < 0.01 else "  WARN: large difference, check dtype handling")

        size_mb_p = os.path.getsize(prefill_path) / 1024 / 1024
        print(f"ONNX chunk-prefill file size: {size_mb_p:.1f} MB")


if __name__ == "__main__":
    main()
