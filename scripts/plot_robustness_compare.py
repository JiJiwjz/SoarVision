"""
Overlay several models' degradation-robustness tables into one comparison figure —
the headline 答辩 chart (e.g. baseline vs +D1: "D1 keeps mAP flat as fog thickens").

Reads the robustness_table.csv files written by scripts/rfdetr_robustness.py and
plots, per condition (clear -> fog x3 -> lowlight -> noise), one line per model for
three metrics: mAP@50:95, miss-rate (漏检), small-object recall. English labels only
(server fonts lack CJK glyphs).

Usage
-----
    python scripts/plot_robustness_compare.py \
        --run baseline=runs/rfdetr/small_hires896_v2/robustness/robustness_table.csv \
        --run D1=runs/rfdetr/small_d1_896/robustness/robustness_table.csv \
        --out runs/rfdetr/robustness_compare.png --title "small@896: baseline vs +D1"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# nicer x-axis order/labels if present
ORDER = ["clear", "fog_light", "fog_medium", "fog_heavy", "lowlight", "noise"]


def load_table(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: ORDER.index(r["condition"]) if r["condition"] in ORDER else 99)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=CSV",
                    help="repeatable; e.g. baseline=path/to/robustness_table.csv")
    ap.add_argument("--out", type=Path, default=Path("runs/rfdetr/robustness_compare.png"))
    ap.add_argument("--title", default="Degradation robustness")
    args = ap.parse_args()

    series = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        series[label] = load_table(Path(path))
    conds = [r["condition"] for r in next(iter(series.values()))]
    x = range(len(conds))

    panels = [("mAP50_95", "mAP@50:95", False),
              ("miss", "miss-rate (lower=better)", True),
              ("small_recall", "small-object recall", False)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (key, ylab, pct) in zip(axes, panels):
        for label, rows in series.items():
            ys = [float(r[key]) * (100 if pct else 1) for r in rows]
            ax.plot(x, ys, marker="o", ms=5, label=label)
        ax.set_title(ylab)
        ax.set_xticks(list(x)); ax.set_xticklabels(conds, rotation=30, ha="right")
        ax.set_ylabel(ylab); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(args.title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"[compare] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
