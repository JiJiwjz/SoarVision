"""
Offline graded degradation of the test split (plan §2.5: pre-generated & fixed).

For each condition × level it writes degraded images to
    datasets/degraded/<cond>_<level>/images/...
links the original labels, and emits a per-set data.yaml whose `test:` points there,
so scripts/eval.py / eval_robustness.py can evaluate it directly.

Conditions: fog, lowlight, noise. Levels: 1 (mild) → 3 (severe).
A fixed RNG seed per image keeps the degraded set reproducible across runs.

Usage
-----
    python scripts/make_degraded_testset.py                 # all conditions, levels 1-3
    python scripts/make_degraded_testset.py --conditions fog noise --levels 1 2 3
    python scripts/make_degraded_testset.py --split test --limit 500   # quick subset
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from _common import CLASS_NAMES, REPO_ROOT, images_dir, labels_dir, label_path_for, list_images

OUT_ROOT = REPO_ROOT / "datasets" / "degraded"

# Per-level intensity. Tuned to span "noticeable" → "harsh but still annotatable".
FOG_BETA = {1: 0.6, 2: 1.2, 3: 2.0}        # scattering coefficient for I = J*t + A*(1-t)
FOG_AIRLIGHT = 0.85
LOWLIGHT_GAIN = {1: 0.6, 2: 0.4, 3: 0.25}  # multiplicative brightness
NOISE_SIGMA = {1: 10, 2: 25, 3: 45}        # additive Gaussian std (0-255)


def _rng_for(name: str, cond: str, level: int) -> np.random.Generator:
    seed = int(hashlib.md5(f"{name}|{cond}|{level}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def apply_fog(img: np.ndarray, level: int, rng) -> np.ndarray:
    """Physical scattering model with a smooth depth gradient (top = far = foggier)."""
    h, w = img.shape[:2]
    depth = np.linspace(0.2, 1.0, h)[:, None]              # 0..1 top→bottom proxy
    depth = np.repeat(depth, w, axis=1)
    t = np.exp(-FOG_BETA[level] * depth)[..., None]        # transmission
    A = FOG_AIRLIGHT * 255.0
    out = img.astype(np.float32) * t + A * (1 - t)
    return out.clip(0, 255).astype(np.uint8)


def apply_lowlight(img: np.ndarray, level: int, rng) -> np.ndarray:
    gain = LOWLIGHT_GAIN[level]
    out = img.astype(np.float32) * gain
    out += rng.normal(0, 4, img.shape)                     # mild read noise in the dark
    return out.clip(0, 255).astype(np.uint8)


def apply_noise(img: np.ndarray, level: int, rng) -> np.ndarray:
    out = img.astype(np.float32) + rng.normal(0, NOISE_SIGMA[level], img.shape)
    return out.clip(0, 255).astype(np.uint8)


APPLY = {"fog": apply_fog, "lowlight": apply_lowlight, "noise": apply_noise}


def write_data_yaml(set_dir: Path) -> Path:
    yaml_path = set_dir / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        f"path: {set_dir}\n"
        f"train: images\n"          # not used; eval reads the `test` key
        f"val: images\n"
        f"test: images\n"
        f"names:\n{names}\n"
    )
    return yaml_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--conditions", nargs="+", default=list(APPLY), choices=list(APPLY))
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--limit", type=int, default=0, help="cap #images (0 = all) for quick runs")
    args = ap.parse_args()

    import cv2

    src_imgs = list_images(images_dir(args.split))
    if args.limit:
        src_imgs = src_imgs[: args.limit]
    src_labels = labels_dir(args.split)
    print(f"[degrade] {len(src_imgs)} source images from split={args.split}")

    for cond in args.conditions:
        for level in args.levels:
            set_dir = OUT_ROOT / f"{cond}_{level}"
            (set_dir / "images").mkdir(parents=True, exist_ok=True)
            (set_dir / "labels").mkdir(parents=True, exist_ok=True)
            for img_path in src_imgs:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                rng = _rng_for(img_path.name, cond, level)
                out = APPLY[cond](img, level, rng)
                cv2.imwrite(str(set_dir / "images" / img_path.name), out)
                # Link/copy the unchanged label.
                lbl = label_path_for(img_path, src_labels)
                dst_lbl = set_dir / "labels" / (img_path.stem + ".txt")
                if lbl.exists() and not dst_lbl.exists():
                    dst_lbl.write_text(lbl.read_text())
            yaml_path = write_data_yaml(set_dir)
            print(f"[degrade] {cond}_{level}: {set_dir}  → {yaml_path.name}")

    print(f"\n[degrade] done. Sets under {OUT_ROOT}")
    print("Evaluate one with: python scripts/eval.py --weights best.pt "
          f"--data {OUT_ROOT}/fog_2/data.yaml --split test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
