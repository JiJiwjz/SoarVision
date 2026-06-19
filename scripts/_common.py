"""Shared helpers for the analysis scripts (paths, classes, GT loading, IoU matching)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "datasets" / "Maritime_Detection_YOLO"
DATA_YAML = DATASET_DIR / "data.yaml"

CLASS_NAMES = ["vessel", "small_boat", "buoy"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# COCO area buckets (px²) — used to bucket GT boxes for the small-object story.
SIZE_BUCKETS = {"small": (0, 32**2), "medium": (32**2, 96**2), "large": (96**2, float("inf"))}


def images_dir(split: str) -> Path:
    return DATASET_DIR / "images" / split


def labels_dir(split: str) -> Path:
    return DATASET_DIR / "labels" / split


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_path_for(image_path: Path, split_labels_dir: Path) -> Path:
    return split_labels_dir / (image_path.stem + ".txt")


def load_gt(label_file: Path, img_w: int, img_h: int) -> np.ndarray:
    """Read a YOLO label file → array of [cls, x1, y1, x2, y2] in pixels. Empty if none."""
    if not label_file.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, w, h = (float(x) for x in parts[:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        rows.append([cls, x1, y1, x2, y2])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a[N,4] and b[M,4] in xyxy. Returns [N, M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def match_preds_to_gt(
    gt: np.ndarray, pred_boxes: np.ndarray, pred_cls: np.ndarray, iou_thr: float = 0.5
):
    """Greedy per-class IoU matching at a single IoU threshold.

    Returns (tp_mask, fp_mask) over predictions and (matched_mask) over GT.
    A prediction is TP if it matches an unused GT of the same class with IoU>=thr.
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt)
    tp = np.zeros(n_pred, dtype=bool)
    gt_matched = np.zeros(n_gt, dtype=bool)
    if n_pred == 0 or n_gt == 0:
        return tp, ~tp, gt_matched

    ious = box_iou(pred_boxes, gt[:, 1:5])
    gt_cls = gt[:, 0]
    order = np.argsort(-pred_boxes[:, 0] * 0)  # stable order; preds assumed conf-sorted upstream
    for pi in range(n_pred):
        same_cls = gt_cls == pred_cls[pi]
        cand = np.where(same_cls & ~gt_matched)[0]
        if len(cand) == 0:
            continue
        best = cand[np.argmax(ious[pi, cand])]
        if ious[pi, best] >= iou_thr:
            tp[pi] = True
            gt_matched[best] = True
    return tp, ~tp, gt_matched


def size_bucket(box_xyxy: np.ndarray) -> str:
    area = max(0.0, (box_xyxy[2] - box_xyxy[0])) * max(0.0, (box_xyxy[3] - box_xyxy[1]))
    for name, (lo, hi) in SIZE_BUCKETS.items():
        if lo <= area < hi:
            return name
    return "large"
