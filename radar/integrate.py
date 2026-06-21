"""
Wire the radar fusion blocks (radar/fusion.py) into an RF-DETR model (A3 integration).

Structure (verified empirically on RFDETRNano):
    rfdetr_model.model.model               -> LWDETR (the nn.Module)
    ...model.transformer.decoder           -> TransformerDecoder (.layers, .d_model)
    ...decoder.layers[i]                    -> TransformerDecoderLayer (self/cross-attn + FFN)

We attach one `RadarCrossAttn` per decoder layer and a shared `RadarEncoder` on the
decoder, then class-patch `TransformerDecoderLayer.forward_post` to apply the radar
block AFTER the image deformable cross-attn. The patch is GUARDED: a layer without a
`radar_block` (or with no radar tokens stashed) behaves exactly as before — so the
patch is global but non-fusion models are unaffected.

REVP is fed out-of-band: call `stash_radar(model, revp)` to encode + cache radar
tokens on the layers before the normal forward; `clear_radar(model)` turns fusion off.
Gate init 0 => the fused model starts identical to the RGB baseline.

Self-test (needs rfdetr + a real image):
    python radar/integrate.py --selftest --image datasets/Maritime_Detection_YOLO/images/test/<x>.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from fusion import RadarCrossAttn, RadarEncoder

_PATCHED = False


def _decoder(rfdetr_model):
    """Return the TransformerDecoder inside an RFDETR* wrapper."""
    for path in ("model.model.transformer.decoder", "model.transformer.decoder"):
        cur = rfdetr_model
        ok = True
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                ok = False
                break
        if ok:
            return cur
    raise AttributeError("could not locate transformer.decoder on the RFDETR model")


def _patch_forward_post() -> None:
    """Class-patch TransformerDecoderLayer.forward_post once (guarded, idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    from rfdetr.models.transformer import TransformerDecoderLayer

    orig = TransformerDecoderLayer.forward_post

    def fused_forward_post(self, tgt, memory, *args, **kwargs):
        out = orig(self, tgt, memory, *args, **kwargs)
        rb = getattr(self, "radar_block", None)
        tok = getattr(self, "_radar_tokens", None)
        if rb is None or tok is None:
            return out                                   # non-fusion / radar-off: unchanged
        qpos = kwargs.get("query_pos", None)
        mask = getattr(self, "_radar_mask", None)
        gate = getattr(self, "_radar_ext_gate", None)
        if isinstance(out, tuple):
            return (rb(out[0], qpos, tok, mask, gate),) + tuple(out[1:])
        return rb(out, qpos, tok, mask, gate)

    fused_forward_post._rgq_patched = True
    TransformerDecoderLayer.forward_post = fused_forward_post
    _PATCHED = True


def attach_radar_fusion(rfdetr_model, d_model: int | None = None, nhead: int = 8,
                        revp_ch: int = 5, fuse_last: int | None = None):
    """Attach RadarEncoder + per-layer gated RadarCrossAttn to an RFDETR model.

    fuse_last: only fuse the last N decoder layers (None = all). Returns the model.
    New params live under decoder.radar_encoder and layer.radar_block (in state_dict,
    trainable). Gate init 0 => identical to baseline until trained."""
    dec = _decoder(rfdetr_model)
    d_model = d_model or dec.d_model
    dec.radar_encoder = RadarEncoder(in_ch=revp_ch, d_model=d_model)
    n = len(dec.layers)
    start = 0 if fuse_last is None else max(0, n - fuse_last)
    for i, layer in enumerate(dec.layers):
        layer.radar_block = RadarCrossAttn(d_model, nhead) if i >= start else None
        layer._radar_tokens = None
        layer._radar_mask = None
        layer._radar_ext_gate = None
    _patch_forward_post()
    return rfdetr_model


def stash_radar(rfdetr_model, revp: torch.Tensor, ext_gate: torch.Tensor | None = None) -> None:
    """Encode the REVP map and cache radar tokens on every fused decoder layer."""
    dec = _decoder(rfdetr_model)
    tokens, mask = dec.radar_encoder(revp)
    for layer in dec.layers:
        if getattr(layer, "radar_block", None) is not None:
            layer._radar_tokens = tokens
            layer._radar_mask = mask
            layer._radar_ext_gate = ext_gate


def clear_radar(rfdetr_model) -> None:
    """Turn fusion off (subsequent forwards == RGB baseline)."""
    dec = _decoder(rfdetr_model)
    for layer in dec.layers:
        if hasattr(layer, "_radar_tokens"):
            layer._radar_tokens = None
            layer._radar_mask = None
            layer._radar_ext_gate = None


def radar_parameters(rfdetr_model):
    """Iterator over the fusion-only parameters (for a separate LR / param group)."""
    dec = _decoder(rfdetr_model)
    yield from dec.radar_encoder.parameters()
    for layer in dec.layers:
        if getattr(layer, "radar_block", None) is not None:
            yield from layer.radar_block.parameters()


# --------------------------------------------------------------------------------------
def _selftest(image: str) -> int:
    import numpy as np
    import rfdetr

    print("[it] loading RFDETRNano ...")
    m = rfdetr.RFDETRNano()
    det_base = m.predict(image, threshold=0.3)
    n_base = len(det_base.xyxy) if det_base.xyxy is not None else 0
    print(f"[it] baseline predict: {n_base} dets")

    attach_radar_fusion(m)
    np_params = sum(p.numel() for p in radar_parameters(m))
    print(f"[it] attached fusion: {np_params/1e6:.2f}M radar params on "
          f"{len(_decoder(m).layers)} decoder layers")

    # 1) radar OFF must be byte-identical to baseline (patch is non-destructive)
    det_off = m.predict(image, threshold=0.3)
    n_off = len(det_off.xyxy) if det_off.xyxy is not None else 0
    same = (n_off == n_base) and (n_base == 0 or np.allclose(
        np.sort(det_off.xyxy.reshape(-1)), np.sort(det_base.xyxy.reshape(-1)), atol=1e-3))
    print(f"[it] radar OFF predict: {n_off} dets  identical_to_baseline={same}")
    assert same, "patch changed baseline output with radar off!"

    # 2) encoder runs on a REVP map and stashes tokens on the fused layers
    revp = torch.rand(1, 5, 135, 240)
    revp[:, 4] = (revp[:, 4] > 0.7).float()
    if torch.cuda.is_available():
        revp = revp.cuda()
        _decoder(m).radar_encoder.cuda()
    stash_radar(m, revp)
    tok_set = [getattr(l, "_radar_tokens", None) is not None for l in _decoder(m).layers]
    print(f"[it] stash_radar: tokens set per layer = {tok_set}")
    assert all(tok_set), "radar tokens not stashed on all fused layers"

    print("[it] OK — fusion attached, radar-off non-destructive, encoder+stash work. "
          "(radar-ON end-to-end forward exercised in the A4 training loop.)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--image", required=True, help="a real image path for the predict-identity check")
    args = ap.parse_args()
    if args.selftest:
        return _selftest(args.image)
    ap.error("pass --selftest --image <path>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
