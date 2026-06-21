"""
Visualize RF-DETR predictions — GT (left) vs prediction panels, so misses/false-
positives are obvious at a glance. Supports TWO models side by side (e.g. baseline vs
+D1) and an arbitrary image dir (e.g. a degraded condition), so you can show
"baseline misses the ship in fog, D1 still finds it" — the qualitative 答辩 figure.

Panels per frame: GT | PRED(model1) [ | PRED(model2) ].

Usage
-----
    # single model on the clean test split
    python scripts/rfdetr_viz.py --weights best.pth --variant small --resolution 896 --num 12

    # baseline vs D1 on the HEAVY-FOG degraded set
    python scripts/rfdetr_viz.py \
        --weights  runs/rfdetr/small_hires896_v2/checkpoint_best_total.pth --variant  small \
        --weights2 runs/rfdetr/small_d1_896/checkpoint_best_total.pth      --variant2 small \
        --resolution 896 --resolution2 896 \
        --images-dir datasets/Maritime_Degraded/fog_heavy/images \
        --labels-dir datasets/Maritime_Degraded/fog_heavy/labels \
        --num 12 --out-dir runs/rfdetr/viz_fog_compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from _common import (
    CLASS_NAMES, images_dir, labels_dir, label_path_for, list_images, load_gt,
)
from rfdetr_eval import VARIANTS, build_model

GT_COLOR = (0, 255, 0)  # green (BGR)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--variant", default="nano", choices=list(VARIANTS))
    p.add_argument("--resolution", type=int, default=None,
                   help="MUST match the checkpoint's training resolution (e.g. 896) — "
                        "a hi-res model at the default res artificially kills small-object detections")
    p.add_argument("--weights2", default=None, help="optional 2nd model -> a 3rd panel (e.g. +D1)")
    p.add_argument("--variant2", default="small", choices=list(VARIANTS))
    p.add_argument("--resolution2", type=int, default=None)
    p.add_argument("--label1", default="PRED", help="caption for model1 panel")
    p.add_argument("--label2", default="PRED2", help="caption for model2 panel")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--images-dir", default=None, help="override: e.g. a degraded condition's images/")
    p.add_argument("--labels-dir", default=None, help="override labels dir (defaults next to --images-dir)")
    p.add_argument("--num", type=int, default=12, help="frames to render (evenly sampled)")
    p.add_argument("--conf", type=float, default=0.3)
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


def annotate_pred(img, det, box_ann, lbl_ann, caption):
    panel = box_ann.annotate(scene=img.copy(), detections=det)
    if len(det) > 0:
        labels = [f"{CLASS_NAMES[int(c)]} {s:.2f}" for c, s in zip(det.class_id, det.confidence)]
        panel = lbl_ann.annotate(scene=panel, detections=det, labels=labels)
    cv2.putText(panel, f"{caption} ({len(det)})", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return panel


def main() -> int:
    args = parse_args()
    import supervision as sv

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.weights).resolve().parent / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.variant, args.weights, args.num_classes, args.resolution)
    model2 = (build_model(args.variant2, args.weights2, args.num_classes, args.resolution2)
              if args.weights2 else None)

    box_ann = sv.BoxAnnotator(thickness=2)
    lbl_ann = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    img_root = Path(args.images_dir) if args.images_dir else images_dir(args.split)
    ldir = Path(args.labels_dir) if args.labels_dir else (
        img_root.parent / "labels" if args.images_dir else labels_dir(args.split))
    imgs = list_images(img_root)
    if not imgs:
        print(f"[viz] no images under {img_root}")
        return 1
    idxs = sorted(set(np.linspace(0, len(imgs) - 1, args.num).astype(int)))
    sel = [imgs[i] for i in idxs]
    print(f"[viz] {len(sel)} frames from {img_root}, conf={args.conf}, 2-model={bool(model2)}")

    for p in sel:
        img = cv2.imread(str(p))
        if img is None:
            continue
        det1 = model.predict(str(p), threshold=args.conf)
        panels = []
        if args.gt:
            gt = load_gt(label_path_for(p, ldir), img.shape[1], img.shape[0])
            gt_panel = draw_gt(img, gt)
            cv2.putText(gt_panel, f"GT ({len(gt)})", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panels.append(gt_panel)
        panels.append(annotate_pred(img, det1, box_ann, lbl_ann, args.label1))
        if model2 is not None:
            det2 = model2.predict(str(p), threshold=args.conf)
            panels.append(annotate_pred(img, det2, box_ann, lbl_ann, args.label2))
        cv2.imwrite(str(out_dir / f"{p.stem}.jpg"), np.hstack(panels))

    print(f"[viz] saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
