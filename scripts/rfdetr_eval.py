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

The core (build_model / run_eval) is importable so rfdetr_robustness.py can load the
model once and evaluate many degraded conditions without reloading.

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
    p.add_argument("--resolution", type=int, default=None,
                   help="MUST match the checkpoint's training resolution (e.g. 896 for a 896-trained "
                        "model) — evaluating a hi-res model at the default res silently butchers small-object recall")
    p.add_argument("--optimize", action="store_true",
                   help="call optimize_for_inference() — can hang on the deformable-attn trace; off by default")
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


def build_model(variant: str, weights: str, num_classes: int = len(CLASS_NAMES),
                resolution: int | None = None, optimize: bool = False):
    """Construct an RF-DETR from a checkpoint. resolution MUST match training res."""
    import rfdetr

    ctor_kwargs = {"pretrain_weights": weights, "num_classes": num_classes}
    if resolution is not None:
        ctor_kwargs["resolution"] = resolution
    model = getattr(rfdetr, VARIANTS[variant])(**ctor_kwargs)
    if optimize:
        try:
            model.optimize_for_inference()
        except Exception as e:  # noqa: BLE001 — optimization is best-effort
            print(f"[eval] optimize_for_inference skipped: {e}")
    return model


def run_eval(model, imgs: list[Path], ldir: Path, num_classes: int = len(CLASS_NAMES),
             conf: float = 0.25, iou: float = 0.5, map_threshold: float = 0.05,
             progress: bool = True) -> dict:
    """Evaluate an already-built model over an explicit image list + label dir.

    Returns the metrics dict (mAP / per_class / recall_by_size / overall). Reusable
    across degraded conditions — the model is NOT reloaded here."""
    nc = num_classes

    coco_gt = {"images": [], "annotations": [],
               "categories": [{"id": c + 1, "name": n} for c, n in enumerate(CLASS_NAMES)]}
    dets: list = []
    ann_id = 1

    tp = np.zeros(nc, dtype=np.int64)
    fp = np.zeros(nc, dtype=np.int64)
    n_gt = np.zeros(nc, dtype=np.int64)
    bucket_gt = {b: 0 for b in ("small", "medium", "large")}
    bucket_hit = {b: 0 for b in ("small", "medium", "large")}

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

        det = model.predict(str(img_path), threshold=map_threshold)
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

        keep = pconf >= conf
        opb, opc, oconf = pb[keep], pc[keep], pconf[keep]
        order = np.argsort(-oconf)
        opb, opc = opb[order], opc[order]
        tp_mask, fp_mask, gt_matched = match_preds_to_gt(gt, opb, opc, iou_thr=iou)
        for c in range(nc):
            tp[c] += int(((opc == c) & tp_mask).sum())
            fp[c] += int(((opc == c) & fp_mask).sum())
            n_gt[c] += int((gt[:, 0] == c).sum())
        for gi in range(len(gt)):
            b = size_bucket(gt[gi, 1:5])
            bucket_gt[b] += 1
            if gt_matched[gi]:
                bucket_hit[b] += 1

        if progress and (idx + 1) % 500 == 0:
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

    tot_tp, tot_fp, tot_gt = int(tp.sum()), int(fp.sum()), int(n_gt.sum())
    overall = {
        "recall": tot_tp / tot_gt if tot_gt else 0.0,
        "miss_rate": 1 - (tot_tp / tot_gt) if tot_gt else 0.0,
        "false_discovery_rate": 1 - (tot_tp / (tot_tp + tot_fp)) if (tot_tp + tot_fp) else 0.0,
    }

    return {
        "conf": conf, "iou": iou, "n_images": len(imgs),
        "mAP": overall_map, "overall": overall,
        "per_class": per_class, "recall_by_size": recall_by_size, "size_gt_counts": bucket_gt,
    }


def print_report(report: dict) -> None:
    m = report["mAP"]
    print("\n=== RF-DETR eval ===")
    print(f"mAP50={m['mAP50']:.4f}  mAP50-95={m['mAP50_95']:.4f}  mAP75={m['mAP75']:.4f}")
    print(f"\n{'class':<12}{'miss%(漏检)':>14}{'误检%':>10}{'recall':>9}{'prec':>8}{'GT':>8}")
    for name, mc in report["per_class"].items():
        print(f"{name:<12}{mc['miss_rate']*100:>14.1f}{mc['false_discovery_rate']*100:>10.1f}"
              f"{mc['recall']*100:>9.1f}{mc['precision']*100:>8.1f}{mc['gt']:>8}")
    print("\nrecall by size bucket:")
    for b, v in report["recall_by_size"].items():
        print(f"  {b:<8}{'n/a' if v is None else f'{v*100:.1f}%':>8}  (GT={report['size_gt_counts'][b]})")


def main() -> int:
    args = parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else Path(args.weights).resolve().parent / "eval"
    save_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.variant, args.weights, args.num_classes, args.resolution, args.optimize)

    imgs = list_images(images_dir(args.split))
    if args.limit:
        imgs = imgs[: args.limit]
    print(f"[eval] {len(imgs)} images, variant={args.variant}, weights={args.weights}")

    report = run_eval(model, imgs, labels_dir(args.split), args.num_classes,
                      args.conf, args.iou, args.map_threshold)
    report = {"weights": args.weights, "variant": args.variant, "split": args.split, **report}

    print_report(report)
    out = save_dir / f"rfdetr_eval_{args.split}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[eval] saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
