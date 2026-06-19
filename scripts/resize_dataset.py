"""
Offline-downscale the dataset to a max side, to un-starve the GPU.

On CPU-limited boxes the data pipeline (decoding 1920x1080 frames + augmenting)
bottlenecks training — the GPU sits idle while ~tens of cores decode huge images.
Since RF-DETR square-resizes every image to its training resolution anyway (544
for nano), keeping full 1080p source is pure waste. Downscaling source frames to
a max side >= the training resolution is ~lossless for training but slashes the
per-image decode/augment cost.

Labels are YOLO-normalized (fractions of image size), so they are copied
UNCHANGED — resizing the pixels does not move the boxes.

Output mirrors the type-first layout (images/<split>, labels/<split>) so
scripts/make_rfdetr_dataset.py --src <out> can consume it directly.

Usage
-----
    python scripts/resize_dataset.py --max-side 640                 # -> datasets/Maritime_Detection_YOLO_640
    python scripts/resize_dataset.py --max-side 1024 --workers 24
"""

from __future__ import annotations

import argparse
import shutil
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from _common import DATASET_DIR, IMG_EXTS, list_images

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    RESAMPLE = Image.LANCZOS

SPLITS = ("train", "val", "test")


def resize_one(task) -> int:
    src_img, dst_img, max_side = task
    try:
        im = Image.open(src_img)
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        scale = max_side / max(w, h)
        if scale < 1.0:  # only downscale; never upscale a smaller frame
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), RESAMPLE)
        im.save(dst_img, quality=95)
        return 1
    except Exception as e:  # noqa: BLE001 — report & skip the rare bad frame
        print(f"[resize] FAILED {src_img}: {e}")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DATASET_DIR))
    ap.add_argument("--out", default=None, help="default: <src>_<max_side>")
    ap.add_argument("--max-side", type=int, default=640, help="longest side after downscale (>= training res)")
    ap.add_argument("--workers", type=int, default=24, help="parallel resize procs (keep <= usable CPU cores)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out) if args.out else src.parent / f"{src.name}_{args.max_side}"

    tasks = []
    for split in SPLITS:
        si, sl = src / "images" / split, src / "labels" / split
        if not si.is_dir():
            continue
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for lf in sl.glob("*.txt"):                       # labels copied unchanged
            shutil.copy2(lf, out / "labels" / split / lf.name)
        for img in list_images(si):
            tasks.append((str(img), str(out / "images" / split / img.name), args.max_side))

    print(f"[resize] {len(tasks)} images -> max_side={args.max_side}, {args.workers} workers -> {out}")
    with Pool(args.workers) as pool:
        done = sum(pool.imap_unordered(resize_one, tasks, chunksize=64))
    print(f"[resize] done {done}/{len(tasks)}")

    src_yaml = src / "data.yaml"
    if src_yaml.exists():
        (out / "data.yaml").write_text(src_yaml.read_text().replace(str(src), str(out)))
    print(f"[resize] dataset ready at {out}")
    print(f"[resize] next: python scripts/make_rfdetr_dataset.py --src {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
