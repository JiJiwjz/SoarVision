"""
WaterScenesFusion — aligned (image, REVP map, YOLO boxes) samples for fusion training.

This is the feed for the RF-DETR radar cross-attention branch. Each item bundles the
RGB image with its image-aligned REVP radar map (radar/revp.py) and the YOLO boxes
already remapped onto our 3 SoarVision classes (radar/waterscenes.REMAP).

torch is imported lazily: the pure data-prep / inspection path (iter_samples) runs
without torch, so we can validate parsing locally before any GPU work. The torch
Dataset is only built when you actually train.

Quick check (numpy path, needs the dataset on disk)::

    python radar/dataset.py --root datasets/WaterScenes --split train --limit 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from revp import RevpNorm, build_from_points
from waterscenes import Frame, list_frames, load_classes, load_frame


@dataclass
class FusionSample:
    """Framework-agnostic sample. image: HxWx3 uint8; revp: CxHrxWr float32;
    boxes: [N,5] = cls, cx, cy, w, h (YOLO-normalised, 3 SoarVision classes)."""

    frame_id: str
    image: np.ndarray
    revp: np.ndarray
    boxes: np.ndarray


def frame_to_sample(
    fr: Frame, downsample: int = 8, norm: RevpNorm | None = None, load_image: bool = True,
) -> FusionSample:
    if load_image and fr.image_path.exists():
        img = np.asarray(Image.open(fr.image_path).convert("RGB"))
        h, w = img.shape[:2]
    else:
        img = np.zeros((0, 0, 3), np.uint8)
        # fall back to nominal WaterScenes resolution so the REVP grid is still valid
        w, h = 1920, 1080
    revp = build_from_points(fr.radar, w, h, downsample=downsample, norm=norm)
    return FusionSample(fr.frame_id, img, revp, fr.boxes)


def iter_samples(
    root: Path, split: str | None = None, downsample: int = 8,
    norm: RevpNorm | None = None, load_image: bool = True, limit: int = 0,
):
    """Lazy generator of FusionSample (no torch). Good for inspection / data prep."""
    ws_classes = load_classes(root)
    ids = list_frames(root, split)
    if limit:
        ids = ids[:limit]
    for fid in ids:
        fr = load_frame(root, fid, ws_classes)
        yield frame_to_sample(fr, downsample=downsample, norm=norm, load_image=load_image)


# --------------------------------------------------------------------------------------
# torch Dataset (lazy import — only needed for training)
# --------------------------------------------------------------------------------------
def build_torch_dataset(
    root: Path, split: str | None = None, downsample: int = 8,
    norm: RevpNorm | None = None, image_transform=None,
):
    """Return a torch Dataset yielding dict(image, revp, boxes, frame_id) as tensors.

    image_transform(np_uint8_hwc) -> tensor lets you plug RF-DETR's own preprocessing
    (square resize / normalise) so the RGB stream matches the detector; if None the
    image is returned as a CxHxW float tensor in [0,1]."""
    import torch
    from torch.utils.data import Dataset

    ws_classes = load_classes(root)
    frame_ids = list_frames(root, split)

    class _WaterScenesFusion(Dataset):
        def __len__(self):
            return len(frame_ids)

        def __getitem__(self, i):
            fr = load_frame(root, frame_ids[i], ws_classes)
            s = frame_to_sample(fr, downsample=downsample, norm=norm, load_image=True)
            if image_transform is not None:
                image = image_transform(s.image)
            else:
                image = torch.from_numpy(s.image).permute(2, 0, 1).float() / 255.0
            return {
                "frame_id": s.frame_id,
                "image": image,
                "revp": torch.from_numpy(s.revp),               # [C,Hr,Wr]
                "boxes": torch.from_numpy(s.boxes),             # [N,5] cls,cx,cy,w,h
            }

    return _WaterScenesFusion()


def fusion_collate(batch: list[dict]) -> dict:
    """Stack images/revp; keep boxes as a list (ragged). For a DataLoader."""
    import torch

    return {
        "frame_id": [b["frame_id"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]),
        "revp": torch.stack([b["revp"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
    }


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", default=None)
    ap.add_argument("--downsample", type=int, default=8)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--no-image", action="store_true", help="skip image decode (radar-only check)")
    args = ap.parse_args()

    n = 0
    for s in iter_samples(args.root, args.split, downsample=args.downsample,
                          load_image=not args.no_image, limit=args.limit):
        occ = float(s.revp[-1].sum())
        cls_counts = np.bincount(s.boxes[:, 0].astype(int), minlength=3) if len(s.boxes) else [0, 0, 0]
        print(f"{s.frame_id}: image={s.image.shape} revp={tuple(s.revp.shape)} "
              f"occupied={int(occ)} boxes={len(s.boxes)} per_class={list(cls_counts)}")
        n += 1
    print(f"[dataset] {n} samples inspected (split={args.split})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
