"""
Mamba-3 model wrapped to match the nanochat GPT interface.

Pure PyTorch implementation — no Triton/mamba_ssm dependency.
Adapted from: https://github.com/VikramLex/mamba3-minimal
  Copyright 2026 Vikram Karlex
  Licensed under the Apache License, Version 2.0
  http://www.apache.org/licenses/LICENSE-2.0

"Mamba-3: Improved Sequence Modeling Using State Space Principles"
Lahoti et al., ICLR 2026. arXiv:2603.15569

Modifications to the original model (see also nanochat/NOTICE):
  [~] A (decay): fixed A_log per head → data-dependent dd_A per token per head
      (-softplus(dd_A) projected from in_proj at each sequence position)
  [+] MIMO rank-R: SISO-only ssd_siso() → generalised ssd_mimo() with rank-R
      outer-product state updates; mimo_rank=1 recovers original SISO behaviour
  [~] Partial RoPE (rope_fraction): RoPE on all d_state dims → only first
      d_state * rope_fraction dims rotated (default 0.5)
  [~] B_bias / C_bias init: 1.0 (paper) → 0.02 (empirically more stable)
  [~] SSD precision: float32-only inputs → internal fp32 cast, bfloat16-safe

Key Mamba-3 innovations over Mamba-2:
  1. Trapezoidal discretization (second-order accurate state update)
  2. Complex-valued SSM via data-dependent RoPE (enables state-tracking)
  3. QK-Normalization on B, C projections
  4. Learnable BC bias (head-specific, init=1)
  5. No short convolution (trapezoidal + bias makes conv1d unnecessary)

SSM scan: two-SSD decomposition (γ term + β term) using the pure-PyTorch
ssd_siso() defined in this file. Cumulative RoPE is computed via cumsum
before the SSD calls, matching mamba3-minimal's approach.

nanochat interface: forward(), generate(), setup_optimizer(),
                   estimate_flops(), num_scaling_params(), init_weights()

────────────────────────────────────────────────────────────────────────────────
Changes from mamba3-minimal (upstream)
────────────────────────────────────────────────────────────────────────────────

Architecture
  [+] MIMO rank-R extension (mimo_rank config, ssd_mimo(), _forward_mimo,
      _step_mimo): generalises the SISO outer-product state update to rank R.
      mimo_rank=1 recovers the original SISO behaviour exactly.
  [+] SwiGLU MLP added to each layer (Mamba3Layer = SSM block + SwiGLU).
      The original minimal implementation had no MLP sublayer.
  [+] Partial RoPE (rope_fraction): only the first split=d_state*rope_fraction
      dims of B/C are rotated. rope_fraction=1.0 gives the original full RoPE.
  [+] is_outproj_norm: optional RMSNorm before out_proj (Nemotron-H style).
  [~] A: fixed learnable scalar per head (A_log parameter, -exp(A_log)) in the
      original → data-dependent per-token per-head (dd_A projected from in_proj,
      -softplus(dd_A)) in nanochat. This is the most significant architectural
      departure from mamba3-minimal; it aligns more closely with the intent of
      the Mamba-3 paper (data-dependent decay) though the minimal reference
      implementation kept A fixed.
  [+] A_floor: configurable lower bound for data-dependent A (clamp max=-A_floor)
      for numerical stability.
  [+] tie_embeddings: optional weight tying between wte and lm_head.
  [+] Vocab padding to multiple of 64 (wte/lm_head allocate a padded size;
      logits are sliced back to vocab_size at the output).

Initialisation
  [~] B_bias / C_bias init changed from paper's 1.0 → 0.02 (matches
      mamba3_official; empirically more stable at lr~3e-3).
  [+] Explicit init_weights() with Xavier-uniform for in_proj, zeros for
      out_proj/MLP down-proj, softplus-inverse for dt_bias, ones for RMSNorm.
  [+] to_empty() + meta-device init pattern: init_weights() re-ties lm_head ↔
      wte after loading because assign=True in load_state_dict breaks the tie.

Inference
  [+] InferenceCache extended for MIMO: k_state/v_state gain an extra R dim
      when mimo_rank > 1.  InferenceCache.alloc() handles both shapes.
  [+] Chunk prefill in generate(): prompt tokens are processed in
      chunk_size-aligned blocks (full-sequence SSD path) rather than one token
      at a time, giving O(L/Q) prefill cost instead of O(L).
  [~] generate() yields tokens one at a time (streaming) rather than returning
      the full sequence, matching nanochat's engine interface.

Training / optimiser
  [+] nanochat interface: forward(idx, targets, kv_cache, loss_reduction),
      setup_optimizer(), estimate_flops(), num_scaling_params().
  [+] Detailed parameter groups with separate LRs:
        ssm_proj (in/out_proj)   — weight_decay=0, ssm_lr
        ssm_bias (B/C_bias)      — weight_decay=0, ssm_lr
        ssm_norm (RMSNorm)       — weight_decay=0, ssm_lr
        ssm_dyn  (D, dt_bias)    — weight_decay=0, ssm_lr × 0.1
        mlp                      — weight_decay=wd, matrix_lr
        embedding / lm_head      — separate LRs
  [+] MuonAdamW / DistMuonAdamW optimizer support (DDP-aware).
  [+] uniform_lr mode: bypasses per-group LRs for paper-style training.
  [+] COMPUTE_DTYPE integration: fp32 master-weight Linear, auto bfloat16/fp32
      selection based on hardware (via nanochat.common.COMPUTE_DTYPE).
  [+] estimate_flops(): accounts for MIMO rank R and two-SSD decomposition.
  [+] num_scaling_params(): reports wte, lm_head, matrices, scalars separately.

Numerics
  [~] ssd_siso() / ssd_mimo() cast all inputs to float32 internally to avoid
      overflow in exp/cumsum with bfloat16 inputs; output is cast back to the
      original dtype. The upstream minimal implementation assumed float32 inputs.
  [~] Manual RMSNorm (x * rsqrt(...)) used in generate() decode path and in
      scripts/export_onnx.py instead of F.rms_norm, to avoid aten::rms_norm
      which is unsupported in ONNX opset < 18.
────────────────────────────────────────────────────────────────────────────────
"""

import math
from dataclasses import dataclass
from typing import NamedTuple, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

Device: TypeAlias = str | torch.device | None


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Mamba3Config:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_embd: int = 768       # d_model
    d_state: int = 128      # SSM state dimension N (must be even)
    expand: int = 2         # d_inner = expand * n_embd
    headdim: int = 64       # head dimension P
    chunk_size: int = 64    # SSD chunk size Q
    rope_fraction: float = 0.5   # fraction of d_state dims to apply RoPE (0.5 = first half)
    is_outproj_norm: bool = False # RMSNormGated before out_proj (Nemotron-H style)
    A_floor: float = 1e-4        # lower bound for data-dependent A (clamp max=-A_floor)
    model_arch: str = "mamba3"
    tie_embeddings: bool = True   # share wte ↔ lm_head weights (saves vocab_size × n_embd params)
    mimo_rank: int = 1           # MIMO rank R: 1 = SISO (outer product), >1 = rank-R MIMO

    def __post_init__(self):
        self.d_inner = self.expand * self.n_embd
        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim
        assert self.d_state % 2 == 0, "d_state must be even for complex SSM / RoPE pairing"
        # Partial RoPE: rotate only the first split dims of B, C.
        # split must be even (RoPE works on pairs). rope_fraction=1.0 → full RoPE (legacy).
        split = int(self.d_state * self.rope_fraction)
        if split % 2 != 0:
            split -= 1
        self.split_tensor_size = split          # number of dims that get RoPE
        self.num_rope_angles = split // 2       # number of angle pairs
        # SwiGLU inner dim (Llama convention: ~8/3 * d_model, rounded to 256)
        self.d_mlp_inner = 256 * ((int(2 * (4 * self.n_embd) / 3) + 255) // 256)


# ──────────────────────────────────────────────────────────────────────────────
# Inference cache (replaces KV cache)
# ──────────────────────────────────────────────────────────────────────────────

class InferenceCache(NamedTuple):
    ssm_state: torch.Tensor   # (batch, nheads, headdim, d_state)
    k_state:   torch.Tensor   # (batch, nheads, d_state [, R]) — last K; R dim present for MIMO
    v_state:   torch.Tensor   # (batch, nheads, headdim [, R]) — last V; R dim present for MIMO
    cum_angle: torch.Tensor   # (batch, nheads, num_rope_angles)

    @staticmethod
    def alloc(batch_size: int, cfg: Mamba3Config, device=None):
        R = cfg.mimo_rank
        if R > 1:
            k = torch.zeros(batch_size, cfg.nheads, cfg.d_state, R, device=device)
            v = torch.zeros(batch_size, cfg.nheads, cfg.headdim, R, device=device)
        else:
            k = torch.zeros(batch_size, cfg.nheads, cfg.d_state, device=device)
            v = torch.zeros(batch_size, cfg.nheads, cfg.headdim, device=device)
        return InferenceCache(
            ssm_state=torch.zeros(batch_size, cfg.nheads, cfg.headdim, cfg.d_state, device=device),
            k_state=k,
            v_state=v,
            cum_angle=torch.zeros(batch_size, cfg.nheads, cfg.num_rope_angles, device=device),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Full rotary embedding. Rotates all d_state dimensions.
    x: (..., d_state), angles: (..., d_state // 2)
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    return torch.stack([cos_a * x1 - sin_a * x2,
                        sin_a * x1 + cos_a * x2], dim=-1).flatten(-2)


def segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sum — produces a 1-semiseparable lower-triangular decay mask.
    segsum(x)[..., i, j] = Σ_{k=j+1}^{i} x[..., k]  for i >= j, else -inf
    Used by ssd_mimo to build the intra-chunk decay mask L = exp(segsum(dA)).
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    return x_segsum.masked_fill(~mask, -torch.inf)


def ssd_siso(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    initial_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured State Space Duality — SISO variant with per-token A.

    Supports data-dependent A (B, L, H) unlike mamba_chunk_scan_combined which
    requires a fixed scalar A per head. All internal computation in float32 to
    avoid overflow in exp/cumsum when called with bfloat16 inputs.

    Args:
        x: (B, L, H, P) — pre-scaled SSM input
        A: (B, L, H)    — per-token log-decay (ADT = A * DT, already negative)
        B: (B, L, H, N) — input projection (after bias + RoPE)
        C: (B, L, H, N) — output projection (after bias + RoPE)
        chunk_size: partition size Q

    Returns:
        y:           (B, L, H, P)  — same dtype as input x
        final_state: (B, H, P, N)  — float32
    """
    assert x.shape[1] % chunk_size == 0, (
        f"seqlen ({x.shape[1]}) must be divisible by chunk_size ({chunk_size})"
    )

    out_dtype = x.dtype
    x, A, B, C = x.float(), A.float(), B.float(), C.float()

    x, A, B, C = [rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size)
                  for m in (x, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # Step 1: intra-chunk output (diagonal blocks)
    L = torch.exp(segsum(A))                                         # (B, H, chunks, Q, Q)
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # Step 2: per-chunk states
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # Step 3: inter-chunk recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # Step 4: state-to-output
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y.to(out_dtype), final_state


def ssd_mimo(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    initial_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured State Space Duality — MIMO rank-R variant (Mamba-3 Appendix D).

    Generalises SISO by using rank-R outer-product state updates:
      h_t = α * h_{t-1} + Σ_r B_t[:,r] ⊗ x_t[:,r]    (state: P×N)
      y_t = Σ_{n,r} C_t[n,r] * h_t[:, n]              (output: P)

    State shape is identical to SISO: (batch, nheads, headdim, d_state).
    The rank R is contracted in both the B/x product and the C/h product,
    so the output y has the same shape as SISO (no extra R dimension).

    Args:
        x: (batch, seqlen, nheads, headdim, R) — rank-expanded input (pre-scaled by γ or β)
        A: (batch, seqlen, nheads) — log-decay rates dA = Δ * A (already multiplied)
        B: (batch, seqlen, nheads, d_state, R) — input projection (rank R)
        C: (batch, seqlen, nheads, d_state, R) — output projection (rank R, summed in output)
        chunk_size: partition size Q (seqlen must be divisible by chunk_size)

    Returns:
        y:           (batch, seqlen, nheads, headdim)  — rank R already contracted
        final_state: (batch, nheads, headdim, d_state) — float32
    """
    assert x.shape[1] % chunk_size == 0, (
        f"seqlen ({x.shape[1]}) must be divisible by chunk_size ({chunk_size})"
    )

    out_dtype = x.dtype
    x, A, B, C = x.float(), A.float(), B.float(), C.float()

    x, A, B, C = [rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size)
                  for m in (x, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # Step 1: intra-chunk output.
    # Contracts: n (between C at l and B at s), r (between B and x, shared rank),
    #            q (C's rank, summed independently), s (intra-chunk via decay L).
    # Output: (b, c, l, h, p) — both rank indices contracted.
    L = torch.exp(segsum(A))                                   # (b, h, chunks, Q, Q)
    Y_diag = torch.einsum("bclhnq, bcshnr, bhcls, bcshpr -> bclhp", C, B, L, x)

    # Step 2: per-chunk states (contract input rank r between B and x)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhnr, bhcl, bclhpr -> bchpn", B, decay_states, x)

    # Step 3: inter-chunk SSM recurrence (unchanged from SISO)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # Step 4: state-to-output per chunk.
    # Contracts: n (C's state dim with state), q (C's rank, summed).
    # Output: (b, c, l, h, p) — C rank contracted.
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhnq, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y.to(out_dtype), final_state




# ──────────────────────────────────────────────────────────────────────────────
# Modules
# ──────────────────────────────────────────────────────────────────────────────

class Linear(nn.Linear):
    """fp32 master weights, forward casts to input dtype (matches GPT)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_inner: int):
        super().__init__()
        self.w_gate = Linear(d_model, d_inner, bias=False)
        self.w_up   = Linear(d_model, d_inner, bias=False)
        self.w_down = Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Mamba3Block(nn.Module):
    """Mamba-3 SSM mixer. Supports SISO (mimo_rank=1) and MIMO rank-R (mimo_rank>1)."""

    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg
        R = cfg.mimo_rank

        # RMSNorm always operates on d_state (not d_state*R); applied per-rank in MIMO.
        self.bc_dim = cfg.d_state

        d_in_proj = (
            cfg.d_inner              # z (gating, no rank expansion)
            + cfg.d_inner * R        # x (rank R)
            + 2 * cfg.d_state * R    # B + C (rank R each)
            + 3 * cfg.nheads         # dd_dt + dd_A + trap (trapezoidal)
            + cfg.num_rope_angles    # θ (angles for data-dependent RoPE)
        )
        self.in_proj  = Linear(cfg.n_embd, d_in_proj, bias=False)
        self.out_proj = Linear(cfg.d_inner, cfg.n_embd, bias=False)

        # data-dependent A: computed per-token from in_proj (dd_A), no fixed A_log
        self.D       = nn.Parameter(torch.empty(cfg.nheads))
        self.dt_bias = nn.Parameter(torch.empty(cfg.nheads))

        self.B_norm = RMSNorm(cfg.d_state)
        self.C_norm = RMSNorm(cfg.d_state)

        # Bias shape: (H, N) for SISO, (H, N, R) for MIMO
        if R > 1:
            self.B_bias = nn.Parameter(torch.ones(cfg.nheads, cfg.d_state, R))
            self.C_bias = nn.Parameter(torch.ones(cfg.nheads, cfg.d_state, R))
        else:
            self.B_bias = nn.Parameter(torch.ones(cfg.nheads, cfg.d_state))
            self.C_bias = nn.Parameter(torch.ones(cfg.nheads, cfg.d_state))

    @torch.no_grad()
    def init_weights(self):
        nn.init.ones_(self.D)
        # dt_bias: softplus-inverse of log-uniform(0.001, 0.1) — matches official mamba3 init
        _dt = torch.exp(
            torch.rand(self.cfg.nheads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias.copy_(_dt + torch.log(-torch.expm1(-_dt)))  # softplus_inverse(_dt)
        # B_bias, C_bias: paper recommends 1.0 but empirically unstable at lr~3e-3.
        # Use 0.02 (matching mamba3_official empirically validated init).
        nn.init.constant_(self.B_bias, 0.02)
        nn.init.constant_(self.C_bias, 0.02)
        # RMSNorm weights: must be explicitly set to ones — to_empty() leaves garbage data.
        nn.init.ones_(self.B_norm.weight)
        nn.init.ones_(self.C_norm.weight)
        n = self.cfg.n_embd
        s = (3 ** 0.5) * (n ** -0.5)
        nn.init.uniform_(self.in_proj.weight, -s * 0.4, s * 0.4)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, u: torch.Tensor, h: InferenceCache | None = None,
                initial_state: InferenceCache | None = None):
        if h is not None:
            return self._step(u, h)
        R = self.cfg.mimo_rank
        if R > 1:
            return self._forward_mimo(u, initial_state=initial_state)
        return self._forward_siso(u, initial_state=initial_state)

    def _forward_siso(self, u: torch.Tensor, initial_state: InferenceCache | None = None):
        """SISO forward (mimo_rank=1)."""
        B, L, _ = u.shape
        cfg = self.cfg

        proj = self.in_proj(u)
        # Split: [z, x, B, C, dd_dt, dd_A, trap, θ]
        z, x, Bv, Cv, dd_dt, dd_A, trap, theta = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner, cfg.d_state, cfg.d_state,
             cfg.nheads, cfg.nheads, cfg.nheads, cfg.num_rope_angles],
            dim=-1,
        )

        compute_dtype = u.dtype

        # data-dependent A and dt
        A    = -F.softplus(dd_A.float())                  # (B, L, H) float32, always < 0
        A    = torch.clamp(A, max=-cfg.A_floor)
        dt_f = F.softplus(dd_dt.float() + self.dt_bias)  # (B, L, H) float32
        ADT  = A * dt_f                                   # (B, L, H) float32

        # Trapezoidal coefficients (lam = sigmoid(trap))
        lam   = torch.sigmoid(trap.float())                         # (B, L, H)
        gamma = (lam * dt_f).to(compute_dtype)                     # (B, L, H)
        beta  = ((1.0 - lam) * dt_f).to(compute_dtype)            # (B, L, H)

        # QK-norm
        Bv = self.B_norm(Bv)
        Cv = self.C_norm(Cv)

        # Expand to (B, L, H, N): SISO — B/C shared across all heads
        Bv = rearrange(Bv, "b l n -> b l 1 n").expand(-1, -1, cfg.nheads, -1).contiguous()
        Cv = rearrange(Cv, "b l n -> b l 1 n").expand(-1, -1, cfg.nheads, -1).contiguous()

        # Add B_bias / C_bias: (H, N) broadcast → (B, L, H, N)
        Bv = Bv + self.B_bias.to(compute_dtype)
        Cv = Cv + self.C_bias.to(compute_dtype)

        # Cumulative data-dependent RoPE angles
        # raw_angles: (B, L, H, num_rope_angles) = dt_f * tanh(θ) * π
        raw_angles = (
            dt_f.to(compute_dtype).unsqueeze(-1)                             # (B, L, H, 1)
            * (torch.tanh(theta.to(compute_dtype)) * math.pi).unsqueeze(2)  # (B, L, 1, num_angles)
        )
        # Carry over cumulative angle from previous chunk (chunk prefill continuity)
        if initial_state is not None:
            cum_angles = initial_state.cum_angle.to(compute_dtype).unsqueeze(1) + torch.cumsum(raw_angles, dim=1)
        else:
            cum_angles = torch.cumsum(raw_angles, dim=1)  # (B, L, H, num_rope_angles)

        # Apply partial RoPE to first split dims of B and C
        split = cfg.split_tensor_size
        Bv = torch.cat([apply_rope(Bv[..., :split], cum_angles), Bv[..., split:]], dim=-1)
        Cv = torch.cat([apply_rope(Cv[..., :split], cum_angles), Cv[..., split:]], dim=-1)

        # x: (B, L, H, P) / z: (B, L, H, P)
        x = rearrange(x, "b l (h p) -> b l h p", p=cfg.headdim)
        z = rearrange(z, "b l (h p) -> b l h p", p=cfg.headdim)

        # Two-SSD trapezoidal decomposition
        # γ term: current B and x. Pass ssm_state as initial state for cross-chunk continuity.
        initial_ssm = initial_state.ssm_state.unsqueeze(1) if initial_state is not None else None
        y_gamma, state_gamma = ssd_siso(
            x * rearrange(gamma, "b l h -> b l h 1"),
            ADT, Bv, Cv, cfg.chunk_size,
            initial_states=initial_ssm,
        )
        # β term: previous B and x.
        # At chunk boundary, the "previous" B/x is k_state/v_state from the prior chunk.
        # β term always starts from zero state (h_0 is carried entirely by the γ term).
        if initial_state is not None:
            B_prev = torch.cat([initial_state.k_state.to(compute_dtype).unsqueeze(1), Bv[:, :-1]], dim=1)
            x_prev = torch.cat([initial_state.v_state.to(compute_dtype).unsqueeze(1), x[:, :-1]], dim=1)
        else:
            B_prev = F.pad(Bv[:, :-1], (0, 0, 0, 0, 1, 0))  # (B, L, H, N)
            x_prev = F.pad(x[:, :-1],  (0, 0, 0, 0, 1, 0))  # (B, L, H, P)
        y_beta, state_beta = ssd_siso(
            x_prev * rearrange(beta, "b l h -> b l h 1"),
            ADT, B_prev, Cv, cfg.chunk_size,
        )

        y = y_gamma + y_beta  # (B, L, H, P)

        # D skip connection
        y = y + rearrange(self.D, "h -> 1 1 h 1").to(compute_dtype) * x

        # Z gating
        y = y * F.silu(z.to(compute_dtype))

        # Output projection
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.out_proj(y)

        h_new = InferenceCache(
            ssm_state=(state_gamma + state_beta).to(compute_dtype),
            k_state=Bv[:, -1],           # (B, H, N) — last K for β term in next step
            v_state=x[:, -1],            # (B, H, P) — last V for β term in next step
            cum_angle=cum_angles[:, -1], # (B, H, num_rope_angles)
        )

        return y, h_new

    def _forward_mimo(self, u: torch.Tensor, initial_state: InferenceCache | None = None):
        """MIMO rank-R forward (mimo_rank > 1)."""
        Batch, L, _ = u.shape
        cfg = self.cfg
        R = cfg.mimo_rank

        proj = self.in_proj(u)
        # Split: [z, x, B, C, dd_dt, dd_A, trap, θ]
        # x and B/C are rank-expanded by R
        z, x, Bv, Cv, dd_dt, dd_A, trap, theta = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner * R, cfg.d_state * R, cfg.d_state * R,
             cfg.nheads, cfg.nheads, cfg.nheads, cfg.num_rope_angles],
            dim=-1,
        )

        compute_dtype = u.dtype

        # data-dependent A and dt
        A    = -F.softplus(dd_A.float())
        A    = torch.clamp(A, max=-cfg.A_floor)
        dt_f = F.softplus(dd_dt.float() + self.dt_bias)  # (Batch, L, H)
        ADT  = A * dt_f

        # Trapezoidal coefficients
        lam   = torch.sigmoid(trap.float())
        gamma = (lam * dt_f).to(compute_dtype)       # (Batch, L, H)
        beta  = ((1.0 - lam) * dt_f).to(compute_dtype)

        # QK-norm per rank: reshape to (..., d_state), norm, reshape back
        Bv = rearrange(Bv, "b l (r n) -> (b l r) n", r=R, n=cfg.d_state)
        Bv = self.B_norm(Bv)
        Bv = rearrange(Bv, "(b l r) n -> b l n r", b=Batch, l=L, r=R)
        Cv = rearrange(Cv, "b l (r n) -> (b l r) n", r=R, n=cfg.d_state)
        Cv = self.C_norm(Cv)
        Cv = rearrange(Cv, "(b l r) n -> b l n r", b=Batch, l=L, r=R)

        # Expand to (Batch, L, H, N, R): B/C shared across heads for each rank
        Bv = rearrange(Bv, "b l n r -> b l 1 n r").expand(-1, -1, cfg.nheads, -1, -1).contiguous()
        Cv = rearrange(Cv, "b l n r -> b l 1 n r").expand(-1, -1, cfg.nheads, -1, -1).contiguous()

        # Add B_bias / C_bias: (H, N, R) broadcast → (Batch, L, H, N, R)
        # Scale by 1/R so that Σ_r B[r] ⊗ x[r] has the same magnitude as the SISO outer product.
        Bv = (Bv + self.B_bias.to(compute_dtype)) / R
        Cv = Cv + self.C_bias.to(compute_dtype)

        # Cumulative data-dependent RoPE angles: (Batch, L, H, num_rope_angles)
        raw_angles = (
            dt_f.to(compute_dtype).unsqueeze(-1)
            * (torch.tanh(theta.to(compute_dtype)) * math.pi).unsqueeze(2)
        )
        # Carry over cumulative angle from previous chunk (chunk prefill continuity)
        if initial_state is not None:
            cum_angles = initial_state.cum_angle.to(compute_dtype).unsqueeze(1) + torch.cumsum(raw_angles, dim=1)
        else:
            cum_angles = torch.cumsum(raw_angles, dim=1)  # (Batch, L, H, num_rope_angles)

        # Apply partial RoPE to first split dims of B and C.
        # Bv: (Batch, L, H, N, R) — rotate the N dim, apply same angle to each rank.
        # Strategy: permute R to before N, apply_rope, permute back.
        split = cfg.split_tensor_size
        Bv_left = Bv[..., :split, :]                      # (Batch, L, H, split, R)
        Bv_left = Bv_left.permute(0, 1, 2, 4, 3)          # (Batch, L, H, R, split)
        Bv_left = apply_rope(Bv_left, cum_angles.unsqueeze(3))  # broadcast angles over R
        Bv_left = Bv_left.permute(0, 1, 2, 4, 3)          # (Batch, L, H, split, R)
        Bv = torch.cat([Bv_left, Bv[..., split:, :]], dim=-2)

        Cv_left = Cv[..., :split, :]
        Cv_left = Cv_left.permute(0, 1, 2, 4, 3)
        Cv_left = apply_rope(Cv_left, cum_angles.unsqueeze(3))
        Cv_left = Cv_left.permute(0, 1, 2, 4, 3)
        Cv = torch.cat([Cv_left, Cv[..., split:, :]], dim=-2)

        # x: (Batch, L, H, P, R) / z: (Batch, L, H, P)
        x = rearrange(x, "b l (h p r) -> b l h p r", p=cfg.headdim, r=R)
        z = rearrange(z, "b l (h p) -> b l h p", p=cfg.headdim)

        # Two-SSD MIMO trapezoidal decomposition
        # γ term: current B and x. Pass ssm_state as initial state for cross-chunk continuity.
        initial_ssm = initial_state.ssm_state.unsqueeze(1) if initial_state is not None else None
        x_gamma = x * rearrange(gamma, "b l h -> b l h 1 1")  # (Batch, L, H, P, R)
        y_gamma, state_gamma = ssd_mimo(x_gamma, ADT, Bv, Cv, cfg.chunk_size,
                                        initial_states=initial_ssm)

        # β term: previous B and x.
        # At chunk boundary, prepend k_state/v_state from the prior chunk instead of zeros.
        # β term always starts from zero state (h_0 is carried entirely by the γ term).
        if initial_state is not None:
            B_prev = torch.cat([initial_state.k_state.to(compute_dtype).unsqueeze(1), Bv[:, :-1]], dim=1)
            x_prev = torch.cat([initial_state.v_state.to(compute_dtype).unsqueeze(1), x[:, :-1]], dim=1)
        else:
            # For 5D tensors (Batch, L, H, N/P, R), pad the L dimension (dim=1)
            B_prev = F.pad(Bv[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))  # (Batch, L, H, N, R)
            x_prev = F.pad(x[:, :-1],  (0, 0, 0, 0, 0, 0, 1, 0))  # (Batch, L, H, P, R)
        x_beta = x_prev * rearrange(beta, "b l h -> b l h 1 1")
        y_beta, state_beta = ssd_mimo(x_beta, ADT, B_prev, Cv, cfg.chunk_size)

        y = (y_gamma + y_beta) / R  # (Batch, L, H, P) — normalize by rank to match SISO scale

        # D skip connection: mean over rank (= sum/R), then apply D
        y = y + rearrange(self.D, "h -> 1 1 h 1").to(compute_dtype) * x.mean(-1)

        # Z gating
        y = y * F.silu(z.to(compute_dtype))

        # Output projection
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.out_proj(y)

        h_new = InferenceCache(
            ssm_state=(state_gamma + state_beta).to(compute_dtype),
            k_state=Bv[:, -1],           # (Batch, H, N, R) — last K
            v_state=x[:, -1],            # (Batch, H, P, R) — last V
            cum_angle=cum_angles[:, -1], # (Batch, H, num_rope_angles)
        )

        return y, h_new

    def _step(self, u: torch.Tensor, h: InferenceCache):
        """Single-token recurrent inference step. Dispatches to SISO or MIMO."""
        R = self.cfg.mimo_rank
        if R > 1:
            return self._step_mimo(u, h)
        return self._step_siso(u, h)

    def _step_siso(self, u: torch.Tensor, h: InferenceCache):
        """Single-token recurrent step for SISO (mimo_rank=1)."""
        cfg = self.cfg

        proj = self.in_proj(u.squeeze(1))
        z, x, Bv, Cv, dd_dt, dd_A, trap, theta = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner, cfg.d_state, cfg.d_state,
             cfg.nheads, cfg.nheads, cfg.nheads, cfg.num_rope_angles],
            dim=-1,
        )

        compute_dtype = u.dtype

        A    = -F.softplus(dd_A.float())
        A    = torch.clamp(A, max=-cfg.A_floor)
        dt_f = F.softplus(dd_dt.float() + self.dt_bias)        # (B, H)
        lam  = torch.sigmoid(trap).to(compute_dtype)
        Bv   = self.B_norm(Bv)
        Cv   = self.C_norm(Cv)

        raw_angle     = dt_f.unsqueeze(-1) * (torch.tanh(theta.float()) * math.pi).unsqueeze(1)
        new_cum_angle = (h.cum_angle.to(compute_dtype) + raw_angle.to(compute_dtype)) % (2 * math.pi)

        dA    = A * dt_f
        alpha = torch.exp(dA).to(compute_dtype)
        beta  = ((1 - lam) * dt_f.to(compute_dtype))
        gamma = (lam * dt_f.to(compute_dtype))

        x = rearrange(x, "b (h p) -> b h p", p=cfg.headdim)

        Bv = Bv.unsqueeze(1) + self.B_bias.to(compute_dtype)  # (B, H, N)
        Cv = Cv.unsqueeze(1) + self.C_bias.to(compute_dtype)  # (B, H, N)

        split = cfg.split_tensor_size
        Bv = torch.cat([apply_rope(Bv[..., :split], new_cum_angle), Bv[..., split:]], dim=-1)
        Cv = torch.cat([apply_rope(Cv[..., :split], new_cum_angle), Cv[..., split:]], dim=-1)

        Bx = torch.einsum("bhn, bhp -> bhpn", Bv, x)
        prev_Bx = torch.einsum("bhn, bhp -> bhpn", h.k_state.to(compute_dtype),
                                                    h.v_state.to(compute_dtype))
        new_state = (
            h.ssm_state.to(compute_dtype) * rearrange(alpha, "b h -> b h 1 1")
            + prev_Bx    * rearrange(beta,  "b h -> b h 1 1")
            + Bx         * rearrange(gamma, "b h -> b h 1 1")
        )

        y = torch.einsum("bhpn, bhn -> bhp", new_state, Cv)
        y = y + rearrange(self.D.to(compute_dtype), "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = y * F.silu(z)
        y = self.out_proj(y)

        h_new = InferenceCache(
            ssm_state=new_state,
            k_state=Bv.squeeze(1),
            v_state=x,
            cum_angle=new_cum_angle,
        )

        return y.unsqueeze(1), h_new

    def _step_mimo(self, u: torch.Tensor, h: InferenceCache):
        """Single-token recurrent step for MIMO rank-R (mimo_rank > 1)."""
        cfg = self.cfg
        R = cfg.mimo_rank
        B = u.shape[0]

        proj = self.in_proj(u.squeeze(1))
        z, x, Bv, Cv, dd_dt, dd_A, trap, theta = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner * R, cfg.d_state * R, cfg.d_state * R,
             cfg.nheads, cfg.nheads, cfg.nheads, cfg.num_rope_angles],
            dim=-1,
        )

        compute_dtype = u.dtype

        A    = -F.softplus(dd_A.float())
        A    = torch.clamp(A, max=-cfg.A_floor)
        dt_f = F.softplus(dd_dt.float() + self.dt_bias)   # (B, H)
        lam  = torch.sigmoid(trap).to(compute_dtype)

        # QK-norm per rank
        Bv = rearrange(Bv, "b (r n) -> (b r) n", r=R, n=cfg.d_state)
        Bv = self.B_norm(Bv)
        Bv = rearrange(Bv, "(b r) n -> b n r", b=B, r=R)
        Cv = rearrange(Cv, "b (r n) -> (b r) n", r=R, n=cfg.d_state)
        Cv = self.C_norm(Cv)
        Cv = rearrange(Cv, "(b r) n -> b n r", b=B, r=R)

        raw_angle     = dt_f.unsqueeze(-1) * (torch.tanh(theta.float()) * math.pi).unsqueeze(1)
        new_cum_angle = (h.cum_angle.to(compute_dtype) + raw_angle.to(compute_dtype)) % (2 * math.pi)

        dA    = A * dt_f
        alpha = torch.exp(dA).to(compute_dtype)
        beta  = ((1 - lam) * dt_f.to(compute_dtype))
        gamma = (lam * dt_f.to(compute_dtype))

        x = rearrange(x, "b (h p r) -> b h p r", p=cfg.headdim, r=R)

        # Expand B/C to (B, H, N, R) and add bias
        Bv = Bv.unsqueeze(1).expand(-1, cfg.nheads, -1, -1).contiguous()  # (B, H, N, R)
        Cv = Cv.unsqueeze(1).expand(-1, cfg.nheads, -1, -1).contiguous()  # (B, H, N, R)
        Bv = (Bv + self.B_bias.to(compute_dtype)) / R
        Cv = Cv + self.C_bias.to(compute_dtype)

        # Partial RoPE: apply to (B, H, N, R) — permute R before N for apply_rope
        split = cfg.split_tensor_size
        Bv_left = Bv[..., :split, :].permute(0, 1, 3, 2)   # (B, H, R, split)
        Bv_left = apply_rope(Bv_left, new_cum_angle.unsqueeze(2))  # broadcast over R
        Bv_left = Bv_left.permute(0, 1, 3, 2)               # (B, H, split, R)
        Bv = torch.cat([Bv_left, Bv[..., split:, :]], dim=-2)

        Cv_left = Cv[..., :split, :].permute(0, 1, 3, 2)
        Cv_left = apply_rope(Cv_left, new_cum_angle.unsqueeze(2))
        Cv_left = Cv_left.permute(0, 1, 3, 2)
        Cv = torch.cat([Cv_left, Cv[..., split:, :]], dim=-2)

        # MIMO state update: contract rank R between B and x
        # Bx[b,h,p,n] = Σ_r B[b,h,n,r] * x[b,h,p,r]
        Bx = torch.einsum("bhnr, bhpr -> bhpn", Bv, x)
        prev_Bx = torch.einsum("bhnr, bhpr -> bhpn",
                               h.k_state.to(compute_dtype),
                               h.v_state.to(compute_dtype))

        new_state = (
            h.ssm_state.to(compute_dtype) * rearrange(alpha, "b h -> b h 1 1")
            + prev_Bx   * rearrange(beta,  "b h -> b h 1 1")
            + Bx        * rearrange(gamma, "b h -> b h 1 1")
        )

        # MIMO output: C_eff[n] = (1/R) Σ_r C[n,r], normalize to match SISO scale
        C_eff = Cv.mean(-1)  # (B, H, N) — mean over rank (= sum/R)
        y = torch.einsum("bhpn, bhn -> bhp", new_state, C_eff)
        y = y + rearrange(self.D.to(compute_dtype), "h -> h 1") * x.mean(-1)  # D skip (mean rank)
        y = rearrange(y, "b h p -> b (h p)")
        y = y * F.silu(z)
        y = self.out_proj(y)

        h_new = InferenceCache(
            ssm_state=new_state,
            k_state=Bv,   # (B, H, N, R)
            v_state=x,    # (B, H, P, R)
            cum_angle=new_cum_angle,
        )

        return y.unsqueeze(1), h_new


class Mamba3Layer(nn.Module):
    """One full layer: pre-norm + Mamba3Block + pre-norm + SwiGLU."""
    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.mixer_norm = RMSNorm(cfg.n_embd)
        self.mixer      = Mamba3Block(cfg)
        self.mlp_norm   = RMSNorm(cfg.n_embd)
        self.mlp        = SwiGLU(cfg.n_embd, cfg.d_mlp_inner)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level model (nanochat interface)
# ──────────────────────────────────────────────────────────────────────────────

class Mamba3Model(nn.Module):
    """
    Mamba-3 language model with the same interface as nanochat's GPT:
      forward(idx, targets=None, kv_cache=None, loss_reduction='mean')
      generate(tokens, max_tokens, temperature, top_k, seed)
      setup_optimizer(...)
      estimate_flops()
      num_scaling_params()
      init_weights()
    """

    def __init__(self, config: Mamba3Config, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config

        padded = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded != config.vocab_size:
            print0(f"Padding vocab_size {config.vocab_size} -> {padded}")

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded, config.n_embd),
            "h":   nn.ModuleList([Mamba3Layer(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

    @torch.no_grad()
    def init_weights(self):
        n = self.config.n_embd
        s = (3 ** 0.5) * (n ** -0.5)

        if self.config.tie_embeddings:
            # Tied weights: single init for shared embedding/lm_head tensor
            nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.02)
        else:
            nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        for layer in self.transformer.h:
            # SSM mixer
            layer.mixer.init_weights()
            # SwiGLU MLP
            nn.init.uniform_(layer.mlp.w_gate.weight, -s * 0.4, s * 0.4)
            nn.init.uniform_(layer.mlp.w_up.weight,   -s * 0.4, s * 0.4)
            nn.init.zeros_(layer.mlp.w_down.weight)
            # RMSNorm pre-norm weights: must be explicitly ones — to_empty() leaves garbage.
            nn.init.ones_(layer.mixer_norm.weight)
            nn.init.ones_(layer.mlp_norm.weight)

        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

        # Re-tie lm_head ↔ wte after to_empty() (meta-device init breaks the tie)
        if self.config.tie_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self) -> float:
        """
        FLOPs per token estimate for Mamba-3. Follows the same convention as GPT:
          - Linear projections: 6 × matmul_params (forward=2, backward=4)
          - SSD scan: analogous to GPT's attention FLOPs (forward × 3 for fwd+bwd)

        Matmul params: in_proj, out_proj, w_gate, w_up, w_down, lm_head.
        Excluded (not matmuls): wte, D, dt_bias,
                                B_bias, C_bias, RMSNorm weights.

        SSD scan derivation (two-SSD decomposition, per token, per layer):
          Y_diag einsum (chunk×heads×d_state): 2 × chunk_size × nheads × d_state
          Y_off + states einsums (2 terms, heads×headdim×d_state each):
                                              4 × nheads × headdim × d_state
          × 2 SSD calls (gamma + beta terms)
          × 3 for backward  ← same ×3 convention as GPT's attention

        Ref: GPT formula uses 12 × n_head × head_dim × effective_seq per layer.
        """
        cfg = self.config

        # Matmul params: all 2D weights except embeddings and SSM scalars
        ssm_non_matmul = set()
        for layer in self.transformer.h:
            m = layer.mixer
            for p in [m.D, m.dt_bias, m.B_bias, m.C_bias,
                      m.B_norm.weight, m.C_norm.weight,
                      layer.mixer_norm.weight, layer.mlp_norm.weight]:
                ssm_non_matmul.add(id(p))

        nparams_matmul = sum(
            p.numel()
            for name, p in self.named_parameters()
            if 'wte' not in name and id(p) not in ssm_non_matmul
        )
        flops_linear = 6 * nparams_matmul

        # SSD scan FLOPs per token (two calls, forward+backward)
        Q, H, N, P, R = cfg.chunk_size, cfg.nheads, cfg.d_state, cfg.headdim, cfg.mimo_rank
        flops_ssd_per_token = (
            3 * cfg.n_layer           # n_layer layers × ×3 for backward
            * 2                       # gamma + beta SSD calls
            * R                       # MIMO rank R scales scan flops (more outer products)
            * (2 * Q * H * N          # Y_diag: contracts chunk_size × d_state
               + 4 * H * P * N)       # Y_off + states: contracts headdim × d_state (×2 terms)
        )

        return flops_linear + flops_ssd_per_token

    def num_scaling_params(self) -> dict:
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        # With weight tying lm_head shares wte's weight, so count as 0 to avoid double-counting
        lm_head = 0 if self.config.tie_embeddings else sum(p.numel() for p in self.lm_head.parameters())
        # SSM projections and MLP weights
        transformer_matrices = sum(
            p.numel()
            for layer in self.transformer.h
            for p in list(layer.mixer.parameters()) + list(layer.mlp.parameters())
            if p.dim() >= 2
        )
        scalars = sum(
            p.numel()
            for layer in self.transformer.h
            for p in list(layer.mixer.parameters()) + list(layer.mlp.parameters())
            if p.dim() < 2
        )
        total = sum(p.numel() for p in self.parameters())
        return {
            'wte': wte,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.008, embedding_lr=0.001,
                        matrix_lr=0.003, weight_decay=0.05,
                        uniform_lr=None, ssm_lr=None, no_muon=True,
                        scalar_lr=None):  # scalar_lr ignored: mamba3 has no scalar param group
        ddp, rank, local_rank, world_size = get_dist_info()
        model_dim = self.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        if uniform_lr is not None:
            # Mamba-2/3 paper training recipe: uniform LR for all params.
            # Bypasses per-group LRs and dmodel_lr_scale.
            print0(f"Uniform LR mode: all param groups → {uniform_lr:.2e}")
            unembedding_lr = uniform_lr
            embedding_lr   = uniform_lr
            matrix_lr      = uniform_lr
            ssm_lr         = uniform_lr
            dmodel_lr_scale = 1.0
        else:
            # Resolve ssm_lr: separate LR for SSM params (matches mamba3_official)
            if ssm_lr is None:
                ssm_lr = matrix_lr
            print0(f"Scaling LRs ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        print0(f"LRs — embedding: {embedding_lr:.2e}  ssm: {ssm_lr:.2e}  matrix(AdamW): {matrix_lr:.2e}")

        # Collect parameter groups
        embedding_params = list(self.transformer.wte.parameters())
        # With weight tying lm_head shares wte's weight — exclude from separate group
        # to avoid "same param in multiple groups" error in the optimizer.
        lm_head_params = ([] if self.config.tie_embeddings
                          else list(self.lm_head.parameters()))

        # Parameter classification — mirrors mamba3_official grouping:
        #
        # [ssm_proj]   in_proj, out_proj               weight_decay=0, ssm_lr
        # [ssm_bias]   B_bias, C_bias (2D bias tensors) weight_decay=0, ssm_lr
        # [ssm_norm]   B_norm.w, C_norm.w, mixer_norm.w weight_decay=0, ssm_lr
        # [ssm_dyn]    D, dt_bias                       weight_decay=0, ssm_lr * 0.1
        # [mlp]        w_gate, w_up, w_down, mlp_norm.w weight_decay=weight_decay, matrix_lr
        #
        # Key: SSM projection matrices must have weight_decay=0 (high WD destroys SSM dynamics).
        # MLP matrices follow the standard GPT recipe with the passed weight_decay.
        _DYNAMICS_NAMES = {"D", "dt_bias"}
        _SSM_PROJ_NAMES = {"in_proj", "out_proj"}
        _SSM_BIAS_NAMES = {"B_bias", "C_bias"}

        ssm_proj_params = []
        ssm_bias_params = []
        ssm_norm_params = []
        ssm_dyn_params  = []
        mlp_params      = []
        for layer in self.transformer.h:
            for name, p in layer.named_parameters():
                # name e.g.: "mixer.in_proj.weight", "mixer.B_bias", "mlp.w_gate.weight"
                parts = name.split(".")
                leaf  = parts[-1]
                # Second segment identifies the submodule (e.g. "in_proj", "B_bias", "mlp")
                sub   = parts[1] if len(parts) >= 2 else parts[0]
                if sub in _SSM_PROJ_NAMES:
                    ssm_proj_params.append(p)
                elif leaf in _SSM_BIAS_NAMES or sub in _SSM_BIAS_NAMES:
                    ssm_bias_params.append(p)
                elif leaf in _DYNAMICS_NAMES:
                    ssm_dyn_params.append(p)
                elif parts[0] == "mlp":
                    mlp_params.append(p)
                else:
                    # mixer_norm.weight, mlp_norm.weight, B_norm.weight, C_norm.weight
                    ssm_norm_params.append(p)

        dynamics_lr = ssm_lr if uniform_lr is not None else ssm_lr * 0.1
        print0(f"SSM LR: {ssm_lr:.2e}  dynamics LR (D/dt_bias): {dynamics_lr:.2e}  MLP LR: {matrix_lr:.2e}")

        all_counted = (
            set(id(p) for p in embedding_params) |
            set(id(p) for p in lm_head_params) |
            set(id(p) for p in ssm_proj_params) |
            set(id(p) for p in ssm_bias_params) |
            set(id(p) for p in ssm_norm_params) |
            set(id(p) for p in ssm_dyn_params)  |
            set(id(p) for p in mlp_params)
        )
        uncounted = [n for n, p in self.named_parameters() if id(p) not in all_counted]
        assert not uncounted, f"Params not assigned to any group: {uncounted}"

        param_groups = [
            dict(kind='adamw', params=lm_head_params,   lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96),  eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params,  lr=embedding_lr   * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            # SSM group — all weight_decay=0 (matches mamba3_official)
            dict(kind='adamw', params=ssm_dyn_params,   lr=dynamics_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
            dict(kind='adamw', params=ssm_norm_params,   lr=ssm_lr,      betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
            dict(kind='adamw', params=ssm_bias_params,   lr=ssm_lr,      betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        ]
        # SSM projection matrices (in_proj, out_proj): weight_decay=0, grouped by shape for compile
        for shape in sorted({p.shape for p in ssm_proj_params}, key=str):
            group_ps = [p for p in ssm_proj_params if p.shape == shape]
            param_groups.append(dict(
                kind='adamw', params=group_ps, lr=ssm_lr,
                betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
            ))
        # MLP matrices: AdamW (Mamba3 paper uses AdamW for all params, no Muon)
        if mlp_params:
            param_groups.append(dict(
                kind='adamw', params=mlp_params, lr=matrix_lr,
                betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx: torch.Tensor, targets=None, kv_cache=None, loss_reduction='mean'):
        """
        idx: (B, T) token indices
        targets: (B, T) for training loss, None for inference
        kv_cache: ignored (Mamba uses recurrent state, not KV cache)
        """
        B, T = idx.size()

        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)

        # SSD requires seqlen divisible by chunk_size; pad if needed during training
        cfg = self.config
        pad = 0
        if T % cfg.chunk_size != 0:
            pad = cfg.chunk_size - (T % cfg.chunk_size)
            x = F.pad(x, (0, 0, 0, pad))

        for layer in self.transformer.h:
            y, _ = layer.mixer(layer.mixer_norm(x))
            x = x + y
            x = x + layer.mlp(layer.mlp_norm(x))

        if pad > 0:
            x = x[:, :T]

        x = F.rms_norm(x, (x.size(-1),))

        logits = self.lm_head(x)[..., :cfg.vocab_size].float()

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        return logits

    @torch.inference_mode()
    def generate(self, tokens: list, max_tokens: int, temperature: float = 1.0,
                 top_k=None, seed: int = 42):
        """
        Streaming autoregressive generation using Mamba-3 recurrent state.
        Uses chunk-prefill for the prompt then O(1) per-token decode.
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        cfg = self.config

        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        # Allocate inference caches for all layers
        h = [InferenceCache.alloc(1, cfg, device=device) for _ in range(cfg.n_layer)]

        def _forward_step(ids_1d: torch.Tensor):
            """Run one token or a chunk through all layers, updating h in-place."""
            nonlocal h
            seqlen = ids_1d.shape[0]
            x = self.transformer.wte(ids_1d.unsqueeze(0)).to(COMPUTE_DTYPE)  # (1, T, d)
            # pad to chunk_size multiple for full-sequence path
            pad = 0
            if seqlen > 1 and seqlen % cfg.chunk_size != 0:
                pad = cfg.chunk_size - (seqlen % cfg.chunk_size)
                x = F.pad(x, (0, 0, 0, pad))
            cache_in = h if seqlen == 1 else [None] * cfg.n_layer
            new_h = []
            for i, layer in enumerate(self.transformer.h):
                y, hi_new = layer.mixer(layer.mixer_norm(x), cache_in[i])
                x = x + y
                x = x + layer.mlp(layer.mlp_norm(x))
                new_h.append(hi_new)
            h = new_h
            if pad > 0:
                x = x[:, :seqlen]
            return x  # (1, T, d_model)

        # Prefill prompt in chunks
        prefix = torch.tensor(tokens, dtype=torch.long, device=device)
        n_chunked = (prefix.shape[0] // cfg.chunk_size) * cfg.chunk_size
        if n_chunked > 0:
            _forward_step(prefix[:n_chunked])
        for i in range(n_chunked, prefix.shape[0]):
            _forward_step(prefix[i:i+1])

        # Decode
        last_token = torch.tensor([tokens[-1]], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            x = _forward_step(last_token)  # (1, 1, d_model)
            x = F.rms_norm(x[:, -1:], (x.size(-1),))
            logits = self.lm_head(x)[0, 0, :cfg.vocab_size].float()
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float('Inf')
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            token = next_id.item()
            yield token
            last_token = next_id
