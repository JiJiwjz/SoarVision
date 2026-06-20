"""
Visualize RF-DETR predictions on sample frames — side-by-side GT (left) vs
prediction (right) panels, so misses/false-positives are obvious at a glance.
Great for eyeballing the small-object problem and for 答辩 result figures.

Samples frames evenly across the split so both SeaShips (big vessels) and
SMD-VIS (small boats / buoys / distant targets) are covered.

Usage
-----
    python scripts/rfdetr_viz.py --weights runs/rfdetr/nano_base640/checkpoint_best_total.pth \
        --variant nano --num 12 --out-dir runs/rfdetr/nano_base640/viz
    python scripts/rfdetr_viz.py --weights best.pth --variant nano --no-gt   # preds only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from _common import (
    CLASS_NAMES, images_dir, labels_dir, label_path_for, list_images, load_gt,
)

VARIANTS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
            "base": "RFDETRBase", "large": "RFDETRLarge"}
GT_COLOR = (0, 255, 0)  # green


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--variant", default="nano", choices=list(VARIANTS))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num", type=int, default=12, help="frames to render (evenly sampled)")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--resolution", type=int, default=None,
                   help="MUST match the checkpoint's training resolution (e.g. 896) — "
                        "predicting a hi-res model at the default res artificially kills "
                        "small-object detections (same trap as rfdetr_eval.py)")
    p.add_argument("--num-classes", type=int, default=len(CLASS_NAMES))
    p.add_argument("--no-gt", action="store_false", dest="gt", help="preds only (no GT panel)")
    p.add_argument("--out-dir", default=None, help="default: <weights>/../viz")
    return p.parse_args()


def draw_gt(img, gt):
    out = img.copy()
    for row in gt:
        c = int(row[0]); x1, y1, x2, y2 = row[1:5].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), GT_COLOR, 2)
        cv2.putText(out, CLASS_NAMES[c], (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1, cv2.LINE_AA)
    return out


def main() -> int:
    args = parse_args()
    import supervision as sv
    import rfdetr

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.weights).resolve().parent / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    ctor_kwargs = {"pretrain_weights": args.weights, "num_classes": args.num_classes}
    if args.resolution is not None:
        ctor_kwargs["resolution"] = args.resolution
    model = getattr(rfdetr, VARIANTS[args.variant])(**ctor_kwargs)
    box_ann = sv.BoxAnnotator(thickness=2)
    lbl_ann = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    imgs = list_images(images_dir(args.split))
    idxs = sorted(set(np.linspace(0, len(imgs) - 1, args.num).astype(int)))
    sel = [imgs[i] for i in idxs]
    ldir = labels_dir(args.split)
    print(f"[viz] {len(sel)} frames from split={args.split}, conf={args.conf}")

    for p in sel:
        img = cv2.imread(str(p))
        if img is None:
            continue
        det = model.predict(str(p), threshold=args.conf)
        pred_panel = box_ann.annotate(scene=img.copy(), detections=det)
        if len(det) > 0:
            labels = [f"{CLASS_NAMES[int(c)]} {s:.2f}" for c, s in zip(det.class_id, det.confidence)]
            pred_panel = lbl_ann.annotate(scene=pred_panel, detections=det, labels=labels)
        cv2.putText(pred_panel, f"PRED ({len(det)})", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if args.gt:
            h, w = img.shape[:2]
            gt = load_gt(label_path_for(p, ldir), w, h)
            gt_panel = draw_gt(img, gt)
            cv2.putText(gt_panel, f"GT ({len(gt)})", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panel = np.hstack([gt_panel, pred_panel])
        else:
            panel = pred_panel

        cv2.imwrite(str(out_dir / f"{p.stem}.jpg"), panel)

    print(f"[viz] saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
