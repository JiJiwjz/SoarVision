"""
Export a trained RF-DETR checkpoint to ONNX — the portable deployment artifact (X1).

ONNX (not a TensorRT engine) is the right thing to ship: a TensorRT engine is
GPU-architecture-specific (an engine built on the 5090, sm_120, will NOT run on the
Jetson Orin Nano, sm_87). So we export ONNX here and build the TensorRT FP16 engine
ON the Jetson from this ONNX (`trtexec --onnx=... --fp16 --saveEngine=...`).

Uses rfdetr's own `model.export(format="onnx")` (handles the deformable-attn custom
symbolic ops). Needs `pip install onnx onnxsim` on the box.

Usage
-----
    python scripts/rfdetr_export.py --weights runs/rfdetr/small_hires896_v2/checkpoint_best_total.pth \
        --variant small --resolution 896 --out-dir runs/rfdetr/export_small
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rfdetr_eval import VARIANTS, build_model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--variant", default="small", choices=list(VARIANTS))
    ap.add_argument("--resolution", type=int, default=None, help="match training resolution (e.g. 896)")
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/rfdetr/export"))
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args.variant, args.weights, args.num_classes, args.resolution)
    print(f"[export] {args.variant}@{args.resolution} -> ONNX in {args.out_dir} (bs={args.batch_size}, opset={args.opset})")

    try:
        model.export(output_dir=str(args.out_dir), format="onnx",
                     batch_size=args.batch_size, opset_version=args.opset)
    except Exception as e:  # noqa: BLE001 — surface the real failure for X1 debugging
        print(f"[export] FAILED: {type(e).__name__}: {e}")
        return 1

    onnx_files = sorted(args.out_dir.glob("*.onnx"))
    if not onnx_files:
        print("[export] no .onnx produced — check logs above")
        return 1
    for f in onnx_files:
        print(f"[export]   {f.name}  {f.stat().st_size / 1e6:.1f} MB")
    print(f"[export] done -> {args.out_dir}")
    print("[export] next (on the Jetson): trtexec --onnx=<file>.onnx --fp16 --saveEngine=model.engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
