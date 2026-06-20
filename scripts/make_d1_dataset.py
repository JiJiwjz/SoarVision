"""
Build a D1 joint clear+degraded training set for RF-DETR — the OFFLINE realisation
of D1 (degradation-robust training). We deliberately do NOT monkey-patch RF-DETR's
internal transforms (fragile, and rfdetr isn't importable off the training box):
instead we materialise a dataset whose TRAIN split = all clear images PLUS a degraded
copy of a fraction of them, so the detector sees both domains. valid/test stay clean
(robustness is measured separately via scripts/rfdetr_robustness.py).

Degradations come from scripts/degrade.py (Koschmieder fog / low-light / sensor noise,
all paper-grounded). Each degraded copy gets ONE random corruption at a random
severity (fog-weighted, since fog is the competition's core condition). Labels are
geometry-invariant under these photometric corruptions, so they are simply copied.

Input  = split-first RF-DETR layout (datasets/maritime_rfdetr_hires/{train,valid,test}/
          {images,labels}/ + data.yaml), as produced by make_rfdetr_dataset.py.
Output = same layout; train/ augmented, valid/ + test/ symlinked to the clean source.

Usage
-----
    python scripts/make_d1_dataset.py \
        --src datasets/maritime_rfdetr_hires --out datasets/maritime_rfdetr_hires_d1 --frac 0.5
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from _common import IMG_EXTS
from degrade import CORRUPTIONS

# corruption menu for D1 copies (fog-weighted; severities matched to the menace level).
DEG_MENU = (
    [("fog", s) for s in (1, 2, 3, 3, 4, 5)] +      # fog over-represented = core condition
    [("lowlight", s) for s in (2, 3, 4)] +
    [("noise", s) for s in (1, 2, 3)] +
    [("sensor_noise", s) for s in (2, 3)]
)


def _link(src: Path, dst: Path) -> None:
    """Symlink dst -> resolved(src) (src may itself be a symlink from make_rfdetr_dataset)."""
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("datasets/maritime_rfdetr_hires"))
    ap.add_argument("--out", type=Path, default=Path("datasets/maritime_rfdetr_hires_d1"))
    ap.add_argument("--frac", type=float, default=0.5, help="fraction of train images to add a degraded copy of")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, out = args.src, args.out
    if not (src / "data.yaml").exists():
        raise SystemExit(f"no data.yaml under {src} — run make_rfdetr_dataset.py first")
    rng = np.random.default_rng(args.seed)
    out.mkdir(parents=True, exist_ok=True)

    # valid/test: symlink the whole split dir to the clean source (eval stays clean)
    for split in ("valid", "test"):
        s = src / split
        if s.exists():
            _link(s, out / split)

    # train: real dirs = clear (symlinked) + degraded copies (materialised)
    ti, tl = out / "train" / "images", out / "train" / "labels"
    ti.mkdir(parents=True, exist_ok=True)
    tl.mkdir(parents=True, exist_ok=True)
    src_imgs = sorted(p for p in (src / "train" / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    src_ldir = src / "train" / "labels"

    n_clear = n_deg = 0
    for i, img in enumerate(src_imgs):
        lbl = src_ldir / (img.stem + ".txt")
        # 1) clear: symlink image + label
        _link(img, ti / img.name)
        if lbl.exists():
            _link(lbl, tl / lbl.name)
        n_clear += 1
        # 2) degraded copy for a fraction
        if rng.random() < args.frac:
            kind, sev = DEG_MENU[int(rng.integers(0, len(DEG_MENU)))]
            arr = CORRUPTIONS[kind](Image.open(img).convert("RGB"), severity=sev, rng=rng)
            stem = f"{img.stem}__d{kind[0]}{sev}"
            Image.fromarray(arr).save(ti / f"{stem}.jpg", quality=95)
            if lbl.exists():
                (tl / f"{stem}.txt").write_text(lbl.read_text())
            n_deg += 1
        if (i + 1) % 2000 == 0:
            print(f"[d1]   {i + 1}/{len(src_imgs)}  (degraded so far: {n_deg})")

    shutil.copyfile(src / "data.yaml", out / "data.yaml")
    print(f"[d1] train = {n_clear} clear + {n_deg} degraded = {n_clear + n_deg} images")
    print(f"[d1] valid/test symlinked to clean source; data.yaml copied")
    print(f"[d1] saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
