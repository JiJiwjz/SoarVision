"""
Evaluate a trained YOLO26 maritime model on a split.

Two passes:
  1. Official `model.val()` — mAP50, mAP50-95, per-class AP, confusion matrix, FPS.
  2. Custom matching pass at a fixed conf — per-class miss-rate (1-recall),
     false-discovery-rate (误检率, 1-precision), and recall-by-size-bucket
     (small/medium/large) which is the practical AP_S proxy for the P2-head story.

Results are printed and written to <save-dir>/eval_<split>.json (+ ultralytics val plots).

Examples
--------
    python scripts/eval.py --weights runs/maritime/e1_p2_26n/weights/best.pt
    python scripts/eval.py --weights best.pt --split test --conf 0.25
    # Robustness eval on an offline-degraded set (see make_degraded_testset.py):
    python scripts/eval.py --weights best.pt --data datasets/degraded/fog_2/data.yaml --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import (
    CLASS_NAMES,
    DATA_YAML,
    images_dir,
    labels_dir,
    label_path_for,
    list_images,
    load_gt,
    match_preds_to_gt,
    size_bucket,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=str(DATA_YAML), help="dataset yaml (override for degraded sets)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=960, help="match the training imgsz")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.25, help="conf threshold for the custom miss/FP pass")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for the custom matching pass")
    p.add_argument("--save-dir", default=None, help="default: alongside the weights")
    p.add_argument("--no-custom", action="store_true", help="skip the size/miss-rate pass (val only)")
    return p.parse_args()


def official_val(weights, data, split, imgsz, batch, device, save_dir):
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(
        data=data, split=split, imgsz=imgsz, batch=batch, device=device,
        plots=True, project=str(save_dir), name="val", exist_ok=True, verbose=False,
    )
    box = metrics.box
    per_class = {}
    for i, ci in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        name = CLASS_NAMES[int(ci)] if int(ci) < len(CLASS_NAMES) else str(int(ci))
        per_class[name] = {
            "precision": float(p), "recall": float(r),
            "AP50": float(ap50), "AP50_95": float(ap),
        }
    speed = {k: float(v) for k, v in metrics.speed.items()}  # ms/img
    total_ms = sum(speed.values())
    return {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "mAP75": float(box.map75),
        "precision_mean": float(box.mp),
        "recall_mean": float(box.mr),
        "per_class": per_class,
        "speed_ms": speed,
        "fps": float(1000.0 / total_ms) if total_ms > 0 else None,
    }


def custom_pass(weights, split, imgsz, device, conf, iou):
    """Per-class miss/FP rates + recall-by-size-bucket via our own matching."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(weights)
    imgs = list_images(images_dir(split))
    ldir = labels_dir(split)

    n_cls = len(CLASS_NAMES)
    tp = np.zeros(n_cls, dtype=np.int64)        # matched preds
    fp = np.zeros(n_cls, dtype=np.int64)        # unmatched preds
    n_gt = np.zeros(n_cls, dtype=np.int64)      # total GT
    bucket_gt = {b: 0 for b in ("small", "medium", "large")}
    bucket_hit = {b: 0 for b in ("small", "medium", "large")}

    results = model.predict(
        source=[str(p) for p in imgs], imgsz=imgsz, device=device, conf=conf,
        stream=True, verbose=False,
    )
    for img_path, res in zip(imgs, results):
        h, w = res.orig_shape
        gt = load_gt(label_path_for(img_path, ldir), w, h)
        if res.boxes is not None and len(res.boxes):
            pb = res.boxes.xyxy.cpu().numpy()
            pc = res.boxes.cls.cpu().numpy().astype(int)
            conf_arr = res.boxes.conf.cpu().numpy()
            order = np.argsort(-conf_arr)        # high-conf first for greedy matching
            pb, pc = pb[order], pc[order]
        else:
            pb = np.zeros((0, 4), dtype=np.float32)
            pc = np.zeros((0,), dtype=int)

        tp_mask, fp_mask, gt_matched = match_preds_to_gt(gt, pb, pc, iou_thr=iou)
        for c in range(n_cls):
            tp[c] += int(((pc == c) & tp_mask).sum())
            fp[c] += int(((pc == c) & fp_mask).sum())
            n_gt[c] += int((gt[:, 0] == c).sum())
        for gi in range(len(gt)):
            b = size_bucket(gt[gi, 1:5])
            bucket_gt[b] += 1
            if gt_matched[gi]:
                bucket_hit[b] += 1

    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        recall = tp[c] / n_gt[c] if n_gt[c] else 0.0
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        per_class[name] = {
            "gt": int(n_gt[c]), "tp": int(tp[c]), "fp": int(fp[c]),
            "recall": float(recall), "miss_rate": float(1 - recall),
            "precision": float(precision), "false_discovery_rate": float(1 - precision),
        }
    recall_by_size = {
        b: (bucket_hit[b] / bucket_gt[b] if bucket_gt[b] else None) for b in bucket_gt
    }
    return {
        "conf": conf, "iou": iou,
        "per_class": per_class,
        "recall_by_size": recall_by_size,
        "size_gt_counts": bucket_gt,
    }


def print_report(report: dict) -> None:
    o = report["official"]
    print("\n=== Official val ===")
    print(f"mAP50={o['mAP50']:.4f}  mAP50-95={o['mAP50_95']:.4f}  "
          f"P={o['precision_mean']:.3f}  R={o['recall_mean']:.3f}  FPS={o['fps']:.1f}")
    print(f"{'class':<12}{'AP50':>8}{'AP50-95':>10}{'P':>8}{'R':>8}")
    for name, m in o["per_class"].items():
        print(f"{name:<12}{m['AP50']:>8.3f}{m['AP50_95']:>10.3f}{m['precision']:>8.3f}{m['recall']:>8.3f}")

    c = report.get("custom")
    if c:
        print(f"\n=== Custom pass (conf={c['conf']}, IoU={c['iou']}) ===")
        print(f"{'class':<12}{'miss%':>8}{'误检%':>8}{'GT':>8}")
        for name, m in c["per_class"].items():
            print(f"{name:<12}{m['miss_rate']*100:>8.1f}{m['false_discovery_rate']*100:>8.1f}{m['gt']:>8}")
        print("recall by size bucket:")
        for b, v in c["recall_by_size"].items():
            print(f"  {b:<8}{'n/a' if v is None else f'{v*100:.1f}%':>8}  (GT={c['size_gt_counts'][b]})")


def main() -> int:
    args = parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else Path(args.weights).resolve().parents[1] / "eval"
    save_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "weights": args.weights, "data": args.data, "split": args.split, "imgsz": args.imgsz,
        "official": official_val(args.weights, args.data, args.split, args.imgsz, args.batch, args.device, save_dir),
    }
    if not args.no_custom:
        report["custom"] = custom_pass(args.weights, args.split, args.imgsz, args.device, args.conf, args.iou)

    print_report(report)
    out = save_dir / f"eval_{args.split}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[eval] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
