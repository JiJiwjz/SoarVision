"""
Failure-case gallery — find the worst test images and show GT vs prediction.

Each image is scored by (#false-negatives + #false-positives) at a fixed conf/IoU.
The worst N are rendered as side-by-side panels (left: GT, right: prediction) with
missed GT boxes highlighted, so you can eyeball where the model breaks (small_boat
at distance, buoy clutter, fog, etc.).

Usage
-----
    python scripts/gallery.py --weights runs/maritime/e1_p2_26n/weights/best.pt --top 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import (
    CLASS_NAMES,
    images_dir,
    labels_dir,
    label_path_for,
    list_images,
    load_gt,
    match_preds_to_gt,
)

# BGR colors per class.
COLORS = [(0, 200, 0), (0, 165, 255), (255, 80, 0)]
MISS_COLOR = (0, 0, 255)   # red for missed GT
FP_COLOR = (0, 0, 255)     # red for false positives


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=960, help="match the training imgsz")
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--top", type=int, default=30, help="number of worst images to render")
    p.add_argument("--out-dir", default=None, help="default: <weights>/../gallery")
    return p.parse_args()


def draw(img, boxes, classes, color_fn, label_fn):
    import cv2

    out = img.copy()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        c = color_fn(i)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.putText(out, label_fn(i), (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return out


def main() -> int:
    args = parse_args()
    import cv2
    from ultralytics import YOLO

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.weights).resolve().parents[1] / "gallery"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    imgs = list_images(images_dir(args.split))
    ldir = labels_dir(args.split)

    scored = []  # (score, img_path, fn_count, fp_count)
    cache = {}
    results = model.predict(source=[str(p) for p in imgs], imgsz=args.imgsz,
                            device=args.device, conf=args.conf, stream=True, verbose=False)
    for img_path, res in zip(imgs, results):
        h, w = res.orig_shape
        gt = load_gt(label_path_for(img_path, ldir), w, h)
        if res.boxes is not None and len(res.boxes):
            pb = res.boxes.xyxy.cpu().numpy()
            pc = res.boxes.cls.cpu().numpy().astype(int)
            order = np.argsort(-res.boxes.conf.cpu().numpy())
            pb, pc = pb[order], pc[order]
        else:
            pb, pc = np.zeros((0, 4), np.float32), np.zeros((0,), int)
        tp_mask, fp_mask, gt_matched = match_preds_to_gt(gt, pb, pc, iou_thr=args.iou)
        fn = int((~gt_matched).sum())
        fp = int(fp_mask.sum())
        score = fn + fp
        if score > 0:
            scored.append((score, img_path, fn, fp))
            cache[img_path] = (gt, pb, pc, fp_mask, gt_matched)

    scored.sort(key=lambda t: -t[0])
    worst = scored[: args.top]
    print(f"[gallery] {len(scored)} imperfect images; rendering top {len(worst)}")

    for rank, (score, img_path, fn, fp) in enumerate(worst):
        gt, pb, pc, fp_mask, gt_matched = cache[img_path]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gt_panel = draw(
            img, gt[:, 1:5], gt[:, 0].astype(int),
            color_fn=lambda i: MISS_COLOR if not gt_matched[i] else COLORS[int(gt[i, 0]) % 3],
            label_fn=lambda i: ("MISS " if not gt_matched[i] else "") + CLASS_NAMES[int(gt[i, 0]) % 3],
        )
        pred_panel = draw(
            img, pb, pc,
            color_fn=lambda i: FP_COLOR if fp_mask[i] else COLORS[int(pc[i]) % 3],
            label_fn=lambda i: ("FP " if fp_mask[i] else "") + CLASS_NAMES[int(pc[i]) % 3],
        )
        cv2.putText(gt_panel, "GT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(pred_panel, f"PRED  FN={fn} FP={fp}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        panel = np.hstack([gt_panel, pred_panel])
        cv2.imwrite(str(out_dir / f"{rank:03d}_score{score}_{img_path.stem}.jpg"), panel)

    print(f"[gallery] saved → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
