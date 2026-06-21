"""
Radar fusion building blocks for RF-DETR (A3 · query-level radar cross-attention).

Two self-contained nn.Modules (depend only on torch, so they unit-test without the
rfdetr package), to be wired into the RF-DETR decoder at the integration step:

  * RadarEncoder   — REVP map [B,5,Hr,Wr] -> radar tokens [B,L,C] + key_padding_mask.
  * RadarCrossAttn — object queries (Q) softly attend to radar tokens (K/V), with a
    ZERO-INITIALISED gate so the fused model starts identical to the RGB baseline and
    can only *add* signal (the "fusion never hurts clear" property; cf. A4 acceptance).

Design (TransCAR-style soft association): we do NOT hard-project radar onto queries by
calibration; the cross-attention learns which radar tokens each query should trust —
robust to mmWave sparsity + elevation ambiguity. Injection point = after the decoder
layer's image deformable cross-attn, before the FFN.

Self-test (needs torch):
    python radar/fusion.py --selftest
"""

from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn


class RadarEncoder(nn.Module):
    """REVP map -> a short sequence of radar tokens (K/V for cross-attention).

    Input  : revp [B, in_ch, Hr, Wr]  (in_ch=5: range/elev/doppler/power/occupancy)
    Output : tokens [B, L, d_model], key_padding_mask [B, L] (True = pad / no radar)
    L = (Hr//stride) * (Wr//stride). A cell is "pad" when no radar point fell in the
    pooled region (occupancy stays 0), so empty water contributes no spurious K/V.
    """

    def __init__(self, in_ch: int = 5, d_model: int = 256, hidden: int = 64, stride: int = 4):
        super().__init__()
        self.occ_idx = in_ch - 1  # occupancy is the last REVP channel
        self.stride = stride
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, d_model, 1),
        )
        self.pos = nn.Parameter(torch.zeros(1, d_model, 1, 1))  # broadcast learnable pos seed

    def forward(self, revp: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.stem(revp) + self.pos                       # [B, C, h, w]
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)                # [B, L, C]
        # pooled occupancy at EXACTLY the token grid (h,w): a token is valid if any
        # radar point fell in its receptive cell. adaptive pool guarantees size match.
        occ = revp[:, self.occ_idx: self.occ_idx + 1]           # [B,1,Hr,Wr]
        occ_pooled = nn.functional.adaptive_max_pool2d(occ, (h, w)).flatten(1)  # [B, L]
        key_padding_mask = occ_pooled <= 0                      # True = empty = pad
        # guard: if a sample has ZERO radar, keep one token unmasked to avoid NaN softmax
        all_pad = key_padding_mask.all(dim=1)
        if all_pad.any():
            key_padding_mask[all_pad, 0] = False
        return tokens, key_padding_mask


class RadarCrossAttn(nn.Module):
    """Queries attend to radar tokens; gated residual (gate init 0 => starts identity)."""

    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # scalar gate, zero-init: fused model == RGB baseline at start, learns to open up
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, tgt: Tensor, query_pos: Tensor | None,
                radar_tokens: Tensor, key_padding_mask: Tensor | None = None,
                ext_gate: Tensor | None = None) -> Tensor:
        q = tgt if query_pos is None else tgt + query_pos
        attn = self.attn(q, radar_tokens, radar_tokens,
                         key_padding_mask=key_padding_mask, need_weights=False)[0]
        g = torch.tanh(self.gate)
        if ext_gate is not None:          # optional degradation-driven gate (per-sample [B,1,1])
            g = g * ext_gate
        # norm the radar contribution, then gated residual → gate=0 is EXACT identity
        # (fused model == RGB baseline at init; "fusion never hurts clear").
        return tgt + g * self.dropout(self.norm(attn))


def _selftest() -> int:
    torch.manual_seed(0)
    B, C, Hr, Wr, NQ = 2, 256, 135, 240, 300
    revp = torch.rand(B, 5, Hr, Wr)
    revp[:, 4] = (revp[:, 4] > 0.7).float()           # sparse occupancy
    enc = RadarEncoder(d_model=C)
    tokens, mask = enc(revp)
    assert tokens.shape[0] == B and tokens.shape[2] == C, tokens.shape
    assert mask.shape == (B, tokens.shape[1]) and mask.dtype == torch.bool
    print(f"[selftest] RadarEncoder: revp{tuple(revp.shape)} -> tokens{tuple(tokens.shape)} "
          f"mask pad-ratio={mask.float().mean():.2f}")

    fuse = RadarCrossAttn(d_model=C)
    tgt = torch.randn(B, NQ, C, requires_grad=True)
    qpos = torch.randn(B, NQ, C)
    out = fuse(tgt, qpos, tokens, mask)
    assert out.shape == tgt.shape
    # gate init 0 (tanh 0 = 0) => output must equal input exactly (identity at start)
    assert torch.allclose(out, tgt, atol=1e-6), "zero-gate fusion must be identity!"
    # gradients flow to the gate after opening it
    fuse.gate.data.fill_(0.5)
    fuse(tgt, qpos, tokens, mask).sum().backward()
    assert fuse.gate.grad is not None and tgt.grad is not None
    # all-empty-radar sample must not NaN
    empty = torch.zeros(1, 5, Hr, Wr)
    t2, m2 = enc(empty)
    o2 = RadarCrossAttn(d_model=C)(torch.randn(1, NQ, C), None, t2, m2)
    assert torch.isfinite(o2).all(), "empty-radar produced NaN/Inf"
    print("[selftest] OK — zero-gate=identity, grads flow, empty-radar safe")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    ap.error("pass --selftest")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
