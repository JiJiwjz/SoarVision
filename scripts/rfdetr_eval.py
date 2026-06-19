"""
Evaluate a trained RF-DETR checkpoint on a split — the official baseline table.

Two metric families, computed our own way so RF-DETR and the YOLO26n baseline are
directly comparable (same _common matching):

  * COCO mAP@50 and mAP@50:95 (via pycocotools) — 检测精度.
  * Per-class miss-rate (1-recall = 漏检率), false-discovery-rate (1-precision =
    误检率) and recall-by-size-bucket (small/medium/large, the small-object story),
    at a fixed operating confidence — via _common.match_preds_to_gt.

Eval runs on the ORIGINAL-resolution split (datasets/Maritime_Detection_YOLO), not
the 640 training copy — the model square-resizes internally, so native frames are
the fair test. class_id from RF-DETR predict is 0-indexed and matches CLASS_NAMES.

Usage
-----
    python scripts/rfdetr_eval.py \
        --weights runs/rfdetr/nano_base640/checkpoint_best_total.pth --variant nano
    python scripts/rfdetr_eval.py --weights best.pth --variant small --split test --conf 0.25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from _common import (
    CLASS_NAMES,
    images_dir,
    labels_dir,
    label_path_for,
    list_images,
    load_gt,
    match_preds_to_gt,
    size_bucket,
)

VARIANTS = {
    "nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
    "base": "RFDETRBase", "large": "RFDETRLarge",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="trained RF-DETR checkpoint .pth")
    p.add_argument("--variant", default="nano", choices=list(VARIANTS))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--conf", type=float, default=0.25, help="operating conf for miss/FP/size metrics")
    p.add_argument("--map-threshold", type=float, default=0.05, help="low conf to collect mAP candidates")
    p.add_argument("--iou", type=float, default=0.5, help="IoU for the custom matching pass")
    p.add_argument("--num-classes", type=int, default=len(CLASS_NAMES))
    p.add_argument("--limit", type=int, default=0, help="cap #images (0=all) for quick checks")
    p.add_argument("--save-dir", default=None, help="default: alongside the weights")
    return p.parse_args()


def coco_map(coco_gt: dict, dets: list) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cg = COCO()
    cg.dataset = coco_gt
    cg.createIndex()
    if not dets:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "mAP75": 0.0}
    cd = cg.loadRes(dets)
    e = COCOeval(cg, cd, "bbox")
    e.evaluate(); e.accumulate(); e.summarize()
    return {"mAP50_95": float(e.stats[0]), "mAP50": float(e.stats[1]), "mAP75": float(e.stats[2])}


def main() -> int:
    args = parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else Path(args.weights).resolve().parent / "eval"
    save_dir.mkdir(parents=True, exist_ok=True)

    import rfdetr

    model = getattr(rfdetr, VARIANTS[args.variant])(
        pretrain_weights=args.weights, num_classes=args.num_classes
    )
    try:
        model.optimize_for_inference()
    except Exception as e:  # noqa: BLE001 — optimization is best-effort
        print(f"[eval] optimize_for_inference skipped: {e}")

    imgs = list_images(images_dir(args.split))
    if args.limit:
        imgs = imgs[: args.limit]
    ldir = labels_dir(args.split)
    nc = args.num_classes

    # COCO accumulators (category ids 1-indexed per COCO convention).
    coco_gt = {"images": [], "annotations": [],
               "categories": [{"id": c + 1, "name": n} for c, n in enumerate(CLASS_NAMES)]}
    dets: list = []
    ann_id = 1

    # Custom operating-point accumulators.
    tp = np.zeros(nc, dtype=np.int64)
    fp = np.zeros(nc, dtype=np.int64)
    n_gt = np.zeros(nc, dtype=np.int64)
    bucket_gt = {b: 0 for b in ("small", "medium", "large")}
    bucket_hit = {b: 0 for b in ("small", "medium", "large")}

    print(f"[eval] {len(imgs)} images, variant={args.variant}, weights={args.weights}")
    for idx, img_path in enumerate(imgs):
        w, h = Image.open(img_path).size
        coco_gt["images"].append({"id": idx, "width": w, "height": h, "file_name": img_path.name})
        gt = load_gt(label_path_for(img_path, ldir), w, h)
        for row in gt:
            c, x1, y1, x2, y2 = row
            coco_gt["annotations"].append({
                "id": ann_id, "image_id": idx, "category_id": int(c) + 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0,
            })
            ann_id += 1

        det = model.predict(str(img_path), threshold=args.map_threshold)
        if det.xyxy is not None and len(det.xyxy):
            pb = np.asarray(det.xyxy, dtype=np.float32)
            pc = np.asarray(det.class_id, dtype=int)
            pconf = np.asarray(det.confidence, dtype=np.float32)
        else:
            pb = np.zeros((0, 4), np.float32); pc = np.zeros((0,), int); pconf = np.zeros((0,), np.float32)

        for k in range(len(pb)):
            x1, y1, x2, y2 = pb[k]
            dets.append({"image_id": idx, "category_id": int(pc[k]) + 1,
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         "score": float(pconf[k])})

        # Operating-point pass: keep conf>=args.conf, high-conf first for greedy match.
        keep = pconf >= args.conf
        opb, opc, oconf = pb[keep], pc[keep], pconf[keep]
        order = np.argsort(-oconf)
        opb, opc = opb[order], opc[order]
        tp_mask, fp_mask, gt_matched = match_preds_to_gt(gt, opb, opc, iou_thr=args.iou)
        for c in range(nc):
            tp[c] += int(((opc == c) & tp_mask).sum())
            fp[c] += int(((opc == c) & fp_mask).sum())
            n_gt[c] += int((gt[:, 0] == c).sum())
        for gi in range(len(gt)):
            b = size_bucket(gt[gi, 1:5])
            bucket_gt[b] += 1
            if gt_matched[gi]:
                bucket_hit[b] += 1

        if (idx + 1) % 500 == 0:
            print(f"[eval]   {idx + 1}/{len(imgs)}")

    overall_map = coco_map(coco_gt, dets)

    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        recall = tp[c] / n_gt[c] if n_gt[c] else 0.0
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        per_class[name] = {
            "gt": int(n_gt[c]), "tp": int(tp[c]), "fp": int(fp[c]),
            "recall": float(recall), "miss_rate": float(1 - recall),
            "precision": float(precision), "false_discovery_rate": float(1 - precision),
        }
    recall_by_size = {b: (bucket_hit[b] / bucket_gt[b] if bucket_gt[b] else None) for b in bucket_gt}

    report = {
        "weights": args.weights, "variant": args.variant, "split": args.split,
        "conf": args.conf, "iou": args.iou,
        "mAP": overall_map,
        "per_class": per_class,
        "recall_by_size": recall_by_size,
        "size_gt_counts": bucket_gt,
    }

    print("\n=== RF-DETR eval ===")
    print(f"mAP50={overall_map['mAP50']:.4f}  mAP50-95={overall_map['mAP50_95']:.4f}  mAP75={overall_map['mAP75']:.4f}")
    print(f"\n{'class':<12}{'miss%(漏检)':>14}{'误检%':>10}{'recall':>9}{'prec':>8}{'GT':>8}")
    for name, m in per_class.items():
        print(f"{name:<12}{m['miss_rate']*100:>14.1f}{m['false_discovery_rate']*100:>10.1f}"
              f"{m['recall']*100:>9.1f}{m['precision']*100:>8.1f}{m['gt']:>8}")
    print("\nrecall by size bucket:")
    for b, v in recall_by_size.items():
        print(f"  {b:<8}{'n/a' if v is None else f'{v*100:.1f}%':>8}  (GT={bucket_gt[b]})")

    out = save_dir / f"rfdetr_eval_{args.split}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[eval] saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
