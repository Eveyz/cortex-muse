#!/usr/bin/env python3
"""train_seed_qat.py — seed-kernel QAT training + .ctxb export (Phase 6).

Single-file PyTorch script. The model mirrors the Rust `CortexModel`
byte-for-byte in structure:

* `BitLinearQAT` — ternary weights via the STE quantizer
  `w_q = round(w / mean|w|).clamp(-1, 1)` with
  `(w_q - w).detach() + w`, and per-token activation quantisation to
  `[-127, 127]` with the same STE pattern. The forward scaling is
  `y = (w_q @ x_q) * act_scale * row_scale`, where `row_scale[r]` is
  the per-row mean |w| — exactly the quantity the Rust
  `BitLinearWeights` stores and the `.ctxb` file carries.
* `DeltaNetLayer` — the Gated DeltaNet delta rule, same maths as
  `crates/cortex-kernels/src/delta_net.rs` (u = S k,
  S = a S + b (v - a u) k^T, o = S q).
* `CortexMemoryBlock` — INT8 ring sliding-window attention (branch A)
  + online hippocampus MLP (branch B). Phase-4 "hippocampus interface
  joint training": the first half of a sequence is written into the
  hippocampus with the *same analytic outer-product rule* the Rust
  `PlasticityWorker` uses (no autograd for the write), the second
  half reads it back; the main LM loss never flows into the
  hippocampus (its read output is detached), mirroring the Rust
  Fast-Path / Plasticity-Path separation.
* `export_to_ctxb` — packs the trained ternary weights into the
  Rust `CacheLine16x16` / `Tile16x4` bit layout
  (`idx = (out % 16) * 4 + (in % 4)`, `00 -> 0, 01 -> +1, 11 -> -1`)
  and writes the `.ctxb` header (magic "CTXB", CRC32 over the
  payload), with tensor names in exactly the order
  `CortexModel::to_ctxb_tensors` writes them.

Self-check (`--self-check`): train 50 steps on random data (or a
local txt file), export a `.ctxb`, run the compiled Rust binary
(`cortex-cli --check-ctxb`) on the same prompt, and compare the
logits. The comparison runs with the hippocampus bypassed on both
sides (the `.ctxb` holds the frozen seed kernel only; the
hippocampus is online memory, not part of the export), and must
satisfy max |logit diff| < 1e-3.

Rounding parity: Rust `f32::round` rounds half *away* from zero,
while `torch.round` / `numpy.round` round half to *even*. Every
quantiser here therefore uses `round_half_away` (sign * floor(|x| +
0.5)), so a 1-LSB activation difference cannot appear between the
two sides.

Usage:
    python train_seed_qat.py --self-check
    python train_seed_qat.py --steps 200 --data book.txt --out seed.ctxb
    python train_seed_qat.py --self-check --rust-bin ./target/release/cortex-cli
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for train_seed_qat.py. "
        "Install the CPU build, e.g.:\n"
        "  python -m venv .venv && .venv/bin/pip install torch "
        "--index-url https://download.pytorch.org/whl/cpu\n"
        f"Original error: {exc}"
    ) from exc


# ---------------------------------------------------------------------------
# Rounding / quantisation helpers (parity with Rust)
# ---------------------------------------------------------------------------

def round_half_away(x: torch.Tensor) -> torch.Tensor:
    """Round half away from zero, matching Rust `f32::round`."""
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def round_half_away_np(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantize_activation(x: torch.Tensor, training: bool):
    """Per-token symmetric INT8 quantisation. Returns (x_q, scale).

    `x` is [..., d]; the scale is computed per trailing vector.
    In training mode the STE `(xq - x).detach() + x` is applied so
    the forward value is the quantised one and the gradient is 1.
    """
    scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    xq = round_half_away(x / scale).clamp(-127, 127)
    if training:
        xq = (xq - x).detach() + x
    return xq, scale


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    d_model: int = 256
    num_macros: int = 2
    num_heads: int = 4
    head_dim: int = 64
    vocab_size: int = 260
    ffn_hidden_dim: int = 512
    hippo_hidden_dim: int = 512
    window_size: int = 512

    @classmethod
    def micro(cls) -> "ModelConfig":
        return cls()

    @classmethod
    def production(cls) -> "ModelConfig":
        return cls(
            d_model=2048, num_macros=6, num_heads=32, head_dim=64,
            vocab_size=32768, ffn_hidden_dim=5504, hippo_hidden_dim=4096,
        )

    @property
    def num_layers(self) -> int:
        return self.num_macros * 4


# ---------------------------------------------------------------------------
# BitLinearQAT
# ---------------------------------------------------------------------------

class BitLinearQAT(nn.Module):
    """1.58-bit linear layer with STE quantisation.

    `weight` is the continuous parameter (float). The forward pass
    uses the ternary quantised weight and per-row scales, exactly as
    the Rust `BitLinearWeights::forward_gemv` does.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / in_features) ** 0.5)

    def quantised_weight(self, training: bool) -> torch.Tensor:
        s = self.weight.abs().mean()
        wq = round_half_away(self.weight / s).clamp(-1, 1)
        if training:
            wq = (wq - self.weight).detach() + self.weight
        return wq

    def row_scales(self) -> torch.Tensor:
        """Per-row mean |w| (detached) — the values exported to .ctxb."""
        return self.weight.detach().abs().mean(dim=1)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        xq, act_scale = quantize_activation(x, training)
        wq = self.quantised_weight(training)
        # y = (wq @ xq) * act_scale * row_scale
        y = xq @ wq.t()
        return y * act_scale * self.row_scales()


# ---------------------------------------------------------------------------
# RMSNorm / SwiGLU / DeltaNet
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + 1e-5) * self.weight


class BitSwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = BitLinearQAT(d_model, hidden_dim)
        self.up_proj = BitLinearQAT(d_model, hidden_dim)
        self.down_proj = BitLinearQAT(hidden_dim, d_model)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x, training)) * self.up_proj(x, training), training)


class DeltaNetLayer(nn.Module):
    """Gated DeltaNet, one layer. State is [B, H, dv, dk] (dv == dk)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, h, hd = cfg.d_model, cfg.num_heads, cfg.head_dim
        self.q_proj = BitLinearQAT(d, h * hd)
        self.k_proj = BitLinearQAT(d, h * hd)
        self.v_proj = BitLinearQAT(d, h * hd)
        self.beta_proj = BitLinearQAT(d, h)
        self.alpha_proj = BitLinearQAT(d, h)
        self.o_proj = BitLinearQAT(h * hd, d)
        self.rms = RMSNorm(d)
        self.log_decay = nn.Parameter(torch.full((h,), -4.23))

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        # x: [B, T, d]; sequential over T (training seqs are short).
        bsz, t, d = x.shape
        h, hd = self.cfg.num_heads, self.cfg.head_dim
        h_norm = self.rms(x)
        q = self.q_proj(h_norm, training).view(bsz, t, h, hd)
        k = self.k_proj(h_norm, training).view(bsz, t, h, hd)
        v = self.v_proj(h_norm, training).view(bsz, t, h, hd)
        beta = torch.sigmoid(self.beta_proj(h_norm, training))          # [B, T, H]
        alpha = torch.exp(-torch.exp(self.log_decay) * F.softplus(self.alpha_proj(h_norm, training)))
        q = F.normalize(q, dim=-1) / math.sqrt(hd)
        k = F.normalize(k, dim=-1)
        state = torch.zeros(bsz, h, hd, hd, device=x.device, dtype=x.dtype)
        outs = []
        for i in range(t):
            qi, ki, vi = q[:, i], k[:, i], v[:, i]          # [B, H, hd]
            bi, ai = beta[:, i].unsqueeze(-1), alpha[:, i].unsqueeze(-1)  # [B, H, 1]
            u = torch.einsum("bhij,bhj->bhi", state, ki)    # S @ k
            delta = vi - ai * u
            state = ai.unsqueeze(-1) * state + bi.unsqueeze(-1) * torch.einsum("bhi,bhj->bhij", delta, ki)
            outs.append(torch.einsum("bhij,bhj->bhi", state, qi))
        o = torch.stack(outs, dim=1).reshape(bsz, t, h * hd)
        return x + self.o_proj(o, training)


# ---------------------------------------------------------------------------
# Hippocampus (analytic outer-product rule, mirroring Rust)
# ---------------------------------------------------------------------------

class HippocampusMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Parameter(torch.randn(hidden_dim, dim) * (1.0 / dim) ** 0.5 * 0.5)
        with torch.no_grad():
            self.w2 = nn.Parameter(torch.randn(dim, hidden_dim) * (1.0 / hidden_dim) ** 0.5 * 0.1)

    def read(self, q: torch.Tensor) -> torch.Tensor:
        return F.silu(q @ self.w1.t()) @ self.w2.t()

    @torch.no_grad()
    def learn_pair(self, k: torch.Tensor, v: torch.Tensor, lr: float = 0.35) -> float:
        """One analytic outer-product update (Rust PlasticityWorker).

        k, v are single vectors (already L2-normalised k). Returns the
        surprise value. This is intentionally *not* autograd: it is
        the same hand-derived rule the Rust engine runs online.
        """
        z = self.w1 @ k
        h = F.silu(z)
        pred = self.w2 @ h
        diff = pred - v
        surprise = 0.5 * (diff.pow(2).mean()).item()
        if surprise < 0.02:
            return surprise
        err = diff / self.dim
        sig = torch.sigmoid(z)
        dsilu = sig * (1.0 + z * (1.0 - sig))
        gz = (self.w2.t() @ err) * dsilu
        self.w2.add_(-lr * torch.outer(err, h))
        self.w1.add_(-lr * torch.outer(gz, k))
        return surprise


class CortexMemoryBlock(nn.Module):
    """Layer 4: ring sliding-window attention + hippocampus + FFN."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, h, hd = cfg.d_model, cfg.num_heads, cfg.head_dim
        self.ring_q = BitLinearQAT(d, h * hd)
        self.ring_k = BitLinearQAT(d, h * hd)
        self.ring_v = BitLinearQAT(d, h * hd)
        self.ring_o = BitLinearQAT(h * hd, d)
        self.ring_rms = RMSNorm(d)
        self.hippo_q = BitLinearQAT(d, d)
        self.hippo_k = BitLinearQAT(d, d)
        self.hippo_v = BitLinearQAT(d, d)
        self.hippo_gate = BitLinearQAT(d, d)
        self.hippo_rms = RMSNorm(d)
        self.hippo = HippocampusMLP(d, cfg.hippo_hidden_dim)
        self.ffn = BitSwiGLU(d, cfg.ffn_hidden_dim)
        self.rms_ffn = RMSNorm(d)
        # Mirrors the Rust `CortexModel.hippo_enabled` flag: when
        # False the hippocampus branch is bypassed (y2 = y1). The
        # .ctxb self-check sets this False on both sides, because the
        # hippocampus is online memory and is not part of the export.
        self.hippo_enabled = True

    def _ring_attention(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        bsz, t, d = x.shape
        h, hd = self.cfg.num_heads, self.cfg.head_dim
        hn = self.ring_rms(x)
        q = self.ring_q(hn, training).view(bsz, t, h, hd)
        k = self.ring_k(hn, training).view(bsz, t, h, hd)
        v = self.ring_v(hn, training).view(bsz, t, h, hd)
        # INT8 KV quantisation (per token) for the check path; in
        # training mode the same quantiser runs through the STE.
        kq, ks = quantize_activation(k, training)
        vq, vs = quantize_activation(v, training)
        kd = kq * ks
        vd = vq * vs
        scores = torch.einsum("bthd,bshd->bhts", q, kd) / math.sqrt(hd)
        pos = torch.arange(t, device=x.device)
        causal = pos[None, :] <= pos[:, None]
        window = (pos[:, None] - pos[None, :]) < self.cfg.window_size
        mask = causal & window
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        o = torch.einsum("bhts,bshd->bthd", attn, vd).reshape(bsz, t, h * hd)
        return x + self.ring_o(o, training)

    def forward(self, x: torch.Tensor, training: bool = False, hippo_train: bool = False) -> torch.Tensor:
        y1 = self._ring_attention(x, training)
        if not self.hippo_enabled:
            y2 = y1
        else:
            # Hippocampus branch.
            hn = self.hippo_rms(y1)
            q = self.hippo_q(hn, training)
            k = self.hippo_k(hn, training)
            v = self.hippo_v(hn, training)
            gate = self.hippo_gate(hn, training)
            qn = F.normalize(q, dim=-1)
            kn = F.normalize(k, dim=-1)
            bsz, t, _ = x.shape
            if hippo_train and training and t >= 2:
                # Phase 4: first half writes, second half reads.
                half = t // 2
                for i in range(half):
                    for b in range(bsz):
                        self.hippo.learn_pair(kn[b, i].detach(), v[b, i].detach())
            with torch.no_grad():
                mem = self.hippo.read(qn)
            y2 = y1 + torch.sigmoid(gate) * mem.detach()
        return y2 + self.ffn(self.rms_ffn(y2), training)


class BitSSMBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.delta = DeltaNetLayer(cfg)
        self.ffn = BitSwiGLU(cfg.d_model, cfg.ffn_hidden_dim)
        self.rms_ffn = RMSNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        y = self.delta(x, training)
        return y + self.ffn(self.rms_ffn(y), training)


class CortexModelTorch(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Parameter(
            torch.randn(cfg.vocab_size, cfg.d_model) * (1.0 / cfg.d_model) ** 0.5
        )
        self.macros = nn.ModuleList()
        for _ in range(cfg.num_macros):
            self.macros.append(
                nn.ModuleList([BitSSMBlock(cfg) for _ in range(3)] + [CortexMemoryBlock(cfg)])
            )
        self.final_rms = RMSNorm(cfg.d_model)

    # -- embedding / LM head (weight-tied, ternary) ----------------------
    def _embedding_matrix(self, training: bool) -> torch.Tensor:
        s = self.embedding.abs().mean()
        wq = round_half_away(self.embedding / s).clamp(-1, 1)
        if training:
            wq = (wq - self.embedding).detach() + self.embedding
        row_scale = self.embedding.detach().abs().mean(dim=1, keepdim=True)
        return wq * row_scale

    def forward(self, ids: torch.Tensor, training: bool = False, hippo_train: bool = False) -> torch.Tensor:
        emb = self._embedding_matrix(training)
        x = emb[ids]
        for macro in self.macros:
            for block in macro:
                if isinstance(block, CortexMemoryBlock):
                    x = block(x, training, hippo_train)
                else:
                    x = block(x, training)
        h = self.final_rms(x)
        return h @ emb.t()

    def loss(self, ids: torch.Tensor, hippo_train: bool = False) -> torch.Tensor:
        logits = self.forward(ids[:, :-1], training=True, hippo_train=hippo_train)
        return F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), ids[:, 1:].reshape(-1))

    @torch.no_grad()
    def quantized_logits(self, ids: list[int]) -> torch.Tensor:
        """Forward with ternary weights + INT8 activations, hippo off.

        Mirrors `cortex-cli --check-ctxb` exactly: the Python layers
        are sequential over T inside one forward call, which is the
        same per-token state evolution as the Rust token-by-token
        loop, and the hippocampus is bypassed on both sides.
        """
        self.eval()
        x = torch.tensor([ids], dtype=torch.long)
        logits = self.forward(x, training=False, hippo_train=False)
        return logits[0, -1]


# ---------------------------------------------------------------------------
# .ctxb export
# ---------------------------------------------------------------------------

def pack_tiles(ternary: np.ndarray) -> bytes:
    """Pack a [rows, cols] int8 ternary matrix into Tile16x4 bytes.

    Layout (must match Rust `BitLinearWeights::from_ternary_matrix`):
    tiles are row-major over (out_block, in_block); inside a tile,
    `idx = (out % 16) * 4 + (in % 4)`, `byte = idx / 4`,
    `shift = (idx % 4) * 2`; codes `01 -> +1, 11 -> -1, 00 -> 0`.
    """
    rows, cols = ternary.shape
    out_blocks = (rows + 15) // 16
    in_blocks = (cols + 3) // 4
    out = bytearray(out_blocks * in_blocks * 16)
    for r in range(rows):
        for c in range(cols):
            val = int(ternary[r, c])
            code = 1 if val == 1 else (3 if val == -1 else 0)
            tile = (r // 16) * in_blocks + (c // 4)
            idx = (r % 16) * 4 + (c % 4)
            out[tile * 16 + idx // 4] |= code << ((idx % 4) * 2)
    return bytes(out)


def ternary_and_scales(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Ternary matrix + per-row scales from a trained float weight."""
    w = weight.detach().cpu().numpy().astype(np.float64)
    s_global = float(np.abs(w).mean())
    tq = round_half_away_np(w / s_global)
    tq = np.clip(tq, -1, 1).astype(np.int8)
    row_scales = np.abs(w).mean(axis=1).astype(np.float32)
    return tq, row_scales


def collect_tensors(model: CortexModelTorch) -> list[tuple[str, int, np.ndarray, np.ndarray | None]]:
    """(name, kind, rows/cols or vector, payload) in Rust order.

    kind 0 = BitLinear (payload = tile bytes, scales = row scales),
    kind 1 = F32Vector (payload = float32 array).
    """
    tensors: list[tuple[str, int, np.ndarray, np.ndarray | None]] = []

    def bitlinear(name: str, layer: BitLinearQAT):
        tq, scales = ternary_and_scales(layer.weight)
        tensors.append((name, 0, tq, scales))

    def vector(name: str, values: torch.Tensor):
        tensors.append((name, 1, values.detach().cpu().numpy().astype(np.float32), None))

    # Embedding (tied LM head): same ternary treatment as a BitLinear.
    tq, scales = ternary_and_scales(model.embedding)
    tensors.append(("embedding", 0, tq, scales))

    for mi, macro in enumerate(model.macros):
        for si, block in enumerate(macro):
            if isinstance(block, BitSSMBlock):
                d = block.delta
                p = f"macro{mi}.ssm{si}.delta"
                bitlinear(f"{p}.q_proj", d.q_proj)
                bitlinear(f"{p}.k_proj", d.k_proj)
                bitlinear(f"{p}.v_proj", d.v_proj)
                bitlinear(f"{p}.beta_proj", d.beta_proj)
                bitlinear(f"{p}.alpha_proj", d.alpha_proj)
                bitlinear(f"{p}.out_proj", d.o_proj)
                p = f"macro{mi}.ssm{si}.ffn"
                bitlinear(f"{p}.gate_proj", block.ffn.gate_proj)
                bitlinear(f"{p}.up_proj", block.ffn.up_proj)
                bitlinear(f"{p}.down_proj", block.ffn.down_proj)
                vector(f"macro{mi}.ssm{si}.rms_ffn", block.rms_ffn.weight)
                vector(f"macro{mi}.ssm{si}.delta.rms", d.rms.weight)
                vector(f"macro{mi}.ssm{si}.delta.log_decay", d.log_decay)
            else:
                p = f"macro{mi}.mem.ring"
                bitlinear(f"{p}.q_proj", block.ring_q)
                bitlinear(f"{p}.k_proj", block.ring_k)
                bitlinear(f"{p}.v_proj", block.ring_v)
                bitlinear(f"{p}.o_proj", block.ring_o)
                vector(f"{p}.rms", block.ring_rms.weight)
                p = f"macro{mi}.mem.hippo"
                bitlinear(f"{p}.q_proj", block.hippo_q)
                bitlinear(f"{p}.k_proj", block.hippo_k)
                bitlinear(f"{p}.v_proj", block.hippo_v)
                bitlinear(f"{p}.gate_proj", block.hippo_gate)
                vector(f"{p}.rms", block.hippo_rms.weight)
                p = f"macro{mi}.mem.ffn"
                bitlinear(f"{p}.gate_proj", block.ffn.gate_proj)
                bitlinear(f"{p}.up_proj", block.ffn.up_proj)
                bitlinear(f"{p}.down_proj", block.ffn.down_proj)
                vector(f"macro{mi}.mem.rms_ffn", block.rms_ffn.weight)
    vector("final_rms", model.final_rms.weight)
    return tensors


def export_to_ctxb(model: CortexModelTorch, filepath: str | Path) -> Path:
    """Write the trained model to a `.ctxb` file (Rust format)."""
    cfg = model.cfg
    tensors = collect_tensors(model)
    payload = bytearray()
    for name, kind, data, scales in tensors:
        name_b = name.encode()
        payload.append(kind)
        payload += struct.pack("<H", len(name_b))
        payload += name_b
        if kind == 0:
            rows, cols = data.shape
            payload += struct.pack("<II", rows, cols)
            payload += np.asarray(scales, dtype="<f4").tobytes()
            tiles = pack_tiles(data)
            payload += struct.pack("<Q", len(tiles))
            payload += tiles
        else:
            payload += struct.pack("<II", len(data), 1)
            payload += struct.pack("<Q", len(data) * 4)
            payload += np.asarray(data, dtype="<f4").tobytes()
    crc = zlib.crc32(bytes(payload)) & 0xFFFFFFFF
    header = struct.pack(
        "<4s7IQI2I12x",
        b"CTXB", 1, cfg.d_model, cfg.num_macros, cfg.num_heads, cfg.head_dim,
        cfg.vocab_size, len(tensors), len(payload), crc,
        cfg.ffn_hidden_dim, cfg.hippo_hidden_dim,
    )
    assert len(header) == 64, len(header)
    path = Path(filepath)
    path.write_bytes(header + bytes(payload))
    return path


# ---------------------------------------------------------------------------
# Data / training
# ---------------------------------------------------------------------------

def load_sequences(data_path: str | None, vocab_size: int, seq_len: int, n_seqs: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    if data_path:
        raw = Path(data_path).read_bytes()
        ids = list(raw)
        seqs = []
        for i in range(n_seqs):
            start = rng.randrange(0, max(1, len(ids) - seq_len))
            seqs.append(ids[start:start + seq_len])
        return torch.tensor(seqs, dtype=torch.long)
    return torch.randint(0, vocab_size, (n_seqs, seq_len), generator=torch.Generator().manual_seed(seed))


def train_seed(
    cfg: ModelConfig,
    steps: int = 50,
    data_path: str | None = None,
    seed: int = 0,
    hippo_train: bool = False,
) -> CortexModelTorch:
    torch.manual_seed(seed)
    model = CortexModelTorch(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    seqs = load_sequences(data_path, cfg.vocab_size, 32, 4, seed)
    for step in range(steps):
        opt.zero_grad()
        loss = model.loss(seqs, hippo_train=hippo_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            print(f"step {step:4d} loss {loss.item():.4f}", flush=True)
    return model


# ---------------------------------------------------------------------------
# Self-check: Rust logits vs PyTorch quantized logits
# ---------------------------------------------------------------------------

def run_self_check(model: CortexModelTorch, ctxb_path: Path, rust_bin: str, prompt: str = "Hello") -> bool:
    ids = list(prompt.encode())
    # PyTorch side: quantised forward, hippocampus bypassed via the
    # flag (same as the Rust side in --check-ctxb mode).
    mem_blocks = [b for macro in model.macros for b in macro if isinstance(b, CortexMemoryBlock)]
    for b in mem_blocks:
        b.hippo_enabled = False
    try:
        py_logits = model.quantized_logits(ids)
    finally:
        for b in mem_blocks:
            b.hippo_enabled = True
    # Rust side.
    cmd = [rust_bin, "--check-ctxb", str(ctxb_path), "--prompt", prompt]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        print("Rust check failed:\n", res.stdout, res.stderr)
        return False
    rust_logits = torch.tensor([float(v) for v in res.stdout.strip().splitlines()])
    if rust_logits.numel() != py_logits.numel():
        print(f"logit count mismatch: rust {rust_logits.numel()} vs py {py_logits.numel()}")
        return False
    err = (rust_logits - py_logits).abs().max().item()
    print(f"self-check: max |logit diff| = {err:.6f} (tolerance 1e-3)")
    return err < 1e-3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cortex-Local seed QAT training + .ctxb export")
    ap.add_argument("--config", choices=["micro", "production"], default="micro")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--data", default=None, help="local txt file for training data (default: random tokens)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="seed.ctxb")
    ap.add_argument("--hippo-train", action="store_true", help="enable phase-4 hippocampus interface joint training")
    ap.add_argument("--self-check", action="store_true", help="train 50 steps, export, and compare logits with the Rust binary")
    ap.add_argument("--rust-bin", default=None, help="path to the cortex-cli binary (default: cargo run)")
    ap.add_argument("--prompt", default="Hello")
    args = ap.parse_args(argv)

    cfg = ModelConfig.micro() if args.config == "micro" else ModelConfig.production()
    steps = 50 if args.self_check else args.steps
    model = train_seed(cfg, steps=steps, data_path=args.data, seed=args.seed, hippo_train=args.hippo_train)
    path = export_to_ctxb(model, args.out)
    print(f"exported {path} ({path.stat().st_size} bytes)")

    if args.self_check:
        rust_bin = args.rust_bin
        if rust_bin is None:
            # Build it once via cargo.
            root = Path(__file__).resolve().parent.parent
            subprocess.run(
                ["cargo", "build", "--release", "-p", "cortex-runtime", "--bin", "cortex-cli"],
                cwd=root, check=True,
            )
            rust_bin = str(root / "target" / "release" / "cortex-cli")
        ok = run_self_check(model, path, rust_bin, prompt=args.prompt)
        print("SELF-CHECK:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
