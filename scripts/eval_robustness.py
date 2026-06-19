"""
Graded-degradation robustness curves (plan §2.5).

Evaluates one or two models on the clean test split plus every offline-degraded
set produced by make_degraded_testset.py, then plots mAP-vs-level per condition.
Pass --weights-d1 to overlay a D1-trained model against the baseline and show the
robustness gain.

Usage
-----
    python scripts/make_degraded_testset.py            # generate the sets first
    python scripts/eval_robustness.py \
        --weights    runs/maritime/e1_p2_26n/weights/best.pt \
        --weights-d1 runs/maritime/e1_p2_26n_d1/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import DATA_YAML, REPO_ROOT

DEGRADED_ROOT = REPO_ROOT / "datasets" / "degraded"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="baseline model")
    p.add_argument("--weights-d1", default=None, help="D1-trained model to overlay (optional)")
    p.add_argument("--conditions", nargs="+", default=["fog", "lowlight", "noise"])
    p.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--imgsz", type=int, default=960, help="match the training imgsz")
    p.add_argument("--device", default="0")
    p.add_argument("--metric", default="mAP50_95", choices=["mAP50", "mAP50_95"])
    p.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "robustness"))
    return p.parse_args()


def run_val(weights, data, imgsz, device) -> dict:
    from ultralytics import YOLO

    m = YOLO(weights).val(data=data, split="test", imgsz=imgsz, device=device,
                          plots=False, verbose=False)
    return {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)}


def collect(weights, conditions, levels, imgsz, device, metric) -> dict:
    """Returns {condition: {0: clean_metric, level: metric, ...}} where level 0 == clean."""
    clean = run_val(weights, str(DATA_YAML), imgsz, device)[metric]
    print(f"[robust] clean {metric}={clean:.4f}")
    curves = {}
    for cond in conditions:
        curves[cond] = {0: clean}
        for lvl in levels:
            yaml_path = DEGRADED_ROOT / f"{cond}_{lvl}" / "data.yaml"
            if not yaml_path.exists():
                print(f"[robust] MISSING {yaml_path} — run make_degraded_testset.py first; skipping")
                continue
            val = run_val(weights, str(yaml_path), imgsz, device)[metric]
            curves[cond][lvl] = val
            print(f"[robust] {cond}_{lvl} {metric}={val:.4f}")
    return curves


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = {"baseline": collect(args.weights, args.conditions, args.levels,
                                  args.imgsz, args.device, args.metric)}
    if args.weights_d1:
        series["D1"] = collect(args.weights_d1, args.conditions, args.levels,
                               args.imgsz, args.device, args.metric)

    (out_dir / "robustness.json").write_text(json.dumps(series, indent=2))

    fig, axes = plt.subplots(1, len(args.conditions), figsize=(5 * len(args.conditions), 4), squeeze=False)
    for ax, cond in zip(axes[0], args.conditions):
        for label, curves in series.items():
            pts = sorted(curves.get(cond, {}).items())
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=label)
        ax.set_title(cond)
        ax.set_xlabel("degradation level (0=clean)")
        ax.set_ylabel(args.metric)
        ax.set_xticks([0] + args.levels)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_curves.png", dpi=140)
    print(f"\n[robust] saved → {out_dir/'robustness_curves.png'} and robustness.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
