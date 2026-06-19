"""
Plot RF-DETR training curves (loss + mAP + per-class AP) from a run's TensorBoard
event file into a single static PNG — for the report / 答辩 figures.

RF-DETR logs a rich scalar set: train/loss, val/loss, val/mAP_50, val/mAP_50_95,
val/AP/<class>, val/recall, etc. We read the largest event file in the run dir.

Usage
-----
    python scripts/plot_curves.py --run-dir runs/rfdetr/nano_base640
    python scripts/plot_curves.py --run-dir runs/rfdetr/nano_hires896 --out curves.png
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import CLASS_NAMES


def load_events(run_dir: Path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = glob.glob(str(run_dir / "*.tfevents*"))
    if not files:
        raise SystemExit(f"no tfevents in {run_dir}")
    f = max(files, key=lambda p: Path(p).stat().st_size)  # the real run = biggest file
    ea = EventAccumulator(f, size_guidance={"scalars": 0})
    ea.Reload()
    return ea, f


def series(ea, tag):
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    evs = ea.Scalars(tag)
    return [e.step for e in evs], [e.value for e in evs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None, help="default: <run-dir>/curves.png")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "curves.png"
    ea, f = load_events(run_dir)
    print(f"[curves] reading {f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (1) loss
    ax = axes[0]
    for tag, lbl in [("train/loss", "train"), ("val/loss", "val")]:
        x, y = series(ea, tag)
        if x:
            ax.plot(x, y, label=lbl)
    ax.set_title("Loss"); ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.grid(alpha=0.3); ax.legend()

    # (2) mAP
    ax = axes[1]
    for tag, lbl in [("val/mAP_50", "mAP@50"), ("val/mAP_50_95", "mAP@50:95"),
                     ("val/ema_mAP_50_95", "EMA mAP@50:95")]:
        x, y = series(ea, tag)
        if x:
            ax.plot(x, y, marker="o", ms=3, label=lbl)
    ax.set_title("Val mAP"); ax.set_xlabel("step"); ax.set_ylabel("mAP")
    ax.grid(alpha=0.3); ax.legend()

    # (3) per-class AP
    ax = axes[2]
    for name in CLASS_NAMES:
        x, y = series(ea, f"val/AP/{name}")
        if x:
            ax.plot(x, y, marker="o", ms=3, label=name)
    ax.set_title("Per-class AP@50:95"); ax.set_xlabel("step"); ax.set_ylabel("AP")
    ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[curves] saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
