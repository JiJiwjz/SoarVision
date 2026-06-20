"""
RF-DETR degradation-robustness table — the headline 雾天鲁棒性 result.

Loads an RF-DETR checkpoint ONCE and evaluates it across the graded degraded
conditions (clear / fog_light / fog_medium / fog_heavy / lowlight / noise) produced
by `scripts/degrade.py`, then emits the per-condition metrics table the competition
asks for (检测精度 mAP / 漏检率 / 误检率 / 实时性). This is the E3-rung evidence
that joint-degradation (D1) training pays off: run it on the baseline and on the
D1-trained model and compare the rows on the degraded conditions.

The degradation physics + citations live in `scripts/degrade.py`; the per-image
metric logic is reused from `rfdetr_eval.run_eval` (identical to the baseline table).

Usage
-----
    # 1) generate the graded test sets once (or pass --gen to do it inline):
    python scripts/degrade.py --gen --split test --out datasets/Maritime_Degraded
    # 2) evaluate across conditions (model loaded once):
    python scripts/rfdetr_robustness.py --weights best.pth --variant small \
        --resolution 896 --degraded-root datasets/Maritime_Degraded --plot --time-n 100
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from _common import IMG_EXTS, list_images
from degrade import TEST_CONDITIONS, generate_testsets
from rfdetr_eval import build_model, run_eval, VARIANTS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--variant", default="small", choices=list(VARIANTS))
    p.add_argument("--resolution", type=int, default=None, help="MUST match training resolution")
    p.add_argument("--degraded-root", type=Path, default=Path("datasets/Maritime_Degraded"))
    p.add_argument("--gen", action="store_true", help="generate the graded sets first (from --split)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="source split for --gen")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--map-threshold", type=float, default=0.05)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="cap #images per condition (0=all) for quick checks")
    p.add_argument("--time-n", type=int, default=0, help="measure inference latency over N images (0=off)")
    p.add_argument("--optimize", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <degraded-root>/robustness")
    p.add_argument("--plot", action="store_true", help="also save a bar chart (needs matplotlib)")
    return p.parse_args()


def discover_conditions(root: Path) -> list[str]:
    """Condition subdirs (each with images/+labels/), ordered by TEST_CONDITIONS first."""
    have = {d.name for d in root.iterdir() if d.is_dir() and (d / "images").is_dir()}
    ordered = [c for c in TEST_CONDITIONS if c in have]
    ordered += sorted(have - set(ordered))
    return ordered


def measure_latency(model, imgs: list[Path], n: int, threshold: float) -> dict:
    sample = imgs[: max(n, 0)]
    if len(sample) < 4:
        return {}
    for p in sample[:3]:                       # warmup
        model.predict(str(p), threshold=threshold)
    t0 = time.perf_counter()
    for p in sample:
        model.predict(str(p), threshold=threshold)
    dt = time.perf_counter() - t0
    ms = dt / len(sample) * 1000.0
    return {"ms_per_img": ms, "fps": 1000.0 / ms, "n_timed": len(sample)}


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or (args.degraded_root / "robustness")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gen:
        from _common import images_dir, labels_dir
        generate_testsets(images_dir(args.split), labels_dir(args.split), args.degraded_root)

    if not args.degraded_root.exists():
        raise SystemExit(f"[robust] {args.degraded_root} missing — run with --gen or run degrade.py --gen first")
    conditions = discover_conditions(args.degraded_root)
    if not conditions:
        raise SystemExit(f"[robust] no condition subdirs under {args.degraded_root}")
    print(f"[robust] conditions: {conditions}")

    model = build_model(args.variant, args.weights, args.num_classes, args.resolution, args.optimize)

    rows, full = [], {}
    latency = {}
    for cond in conditions:
        imgs = list_images(args.degraded_root / cond / "images")
        if args.limit:
            imgs = imgs[: args.limit]
        ldir = args.degraded_root / cond / "labels"
        print(f"\n[robust] === {cond} ({len(imgs)} imgs) ===")
        rep = run_eval(model, imgs, ldir, args.num_classes, args.conf, args.iou, args.map_threshold,
                       progress=False)
        if args.time_n and cond == conditions[0]:
            latency = measure_latency(model, imgs, args.time_n, args.map_threshold)
        full[cond] = rep
        small = rep["recall_by_size"]["small"]
        rows.append({
            "condition": cond, "n_img": rep["n_images"],
            "mAP50": rep["mAP"]["mAP50"], "mAP50_95": rep["mAP"]["mAP50_95"],
            "recall": rep["overall"]["recall"], "miss": rep["overall"]["miss_rate"],
            "fdr": rep["overall"]["false_discovery_rate"],
            "small_recall": (small if small is not None else float("nan")),
        })
        print(f"[robust]   mAP50={rows[-1]['mAP50']:.3f} mAP50:95={rows[-1]['mAP50_95']:.3f} "
              f"漏检={rows[-1]['miss']*100:.1f}% 误检={rows[-1]['fdr']*100:.1f}% "
              f"small_rec={rows[-1]['small_recall']*100:.1f}%")

    # ---- print combined table ----
    print("\n================ Robustness table ================")
    hdr = f"{'condition':<12}{'mAP50':>8}{'mAP50:95':>10}{'recall':>8}{'漏检%':>8}{'误检%':>8}{'small%':>8}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['condition']:<12}{r['mAP50']:>8.3f}{r['mAP50_95']:>10.3f}{r['recall']*100:>8.1f}"
              f"{r['miss']*100:>8.1f}{r['fdr']*100:>8.1f}{r['small_recall']*100:>8.1f}")
    if latency:
        print(f"\n实时性: {latency['ms_per_img']:.1f} ms/img  ({latency['fps']:.1f} FPS, "
              f"n={latency['n_timed']}, variant={args.variant}@{args.resolution})")

    # ---- save CSV + JSON ----
    csv_path = out_dir / "robustness_table.csv"
    with csv_path.open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows)
    json_path = out_dir / "robustness_full.json"
    json_path.write_text(json.dumps(
        {"weights": args.weights, "variant": args.variant, "resolution": args.resolution,
         "latency": latency, "table": rows, "per_condition": full},
        indent=2, ensure_ascii=False))
    print(f"\n[robust] saved -> {csv_path}  +  {json_path}")

    # ---- optional bar chart ----
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            conds = [r["condition"] for r in rows]
            fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
            ax[0].bar(conds, [r["mAP50_95"] for r in rows], color="#3b7dd8")
            ax[0].set_title("mAP@50:95 by condition"); ax[0].set_ylabel("mAP@50:95")
            ax[0].tick_params(axis="x", rotation=30); ax[0].grid(alpha=0.3, axis="y")
            x = range(len(conds))
            ax[1].bar([i - 0.2 for i in x], [r["miss"] * 100 for r in rows], width=0.4, label="漏检%", color="#d8753b")
            ax[1].bar([i + 0.2 for i in x], [r["fdr"] * 100 for r in rows], width=0.4, label="误检%", color="#8e44ad")
            ax[1].set_xticks(list(x)); ax[1].set_xticklabels(conds, rotation=30)
            ax[1].set_title("漏检 / 误检 by condition"); ax[1].set_ylabel("%"); ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")
            fig.tight_layout()
            png = out_dir / "robustness_table.png"
            fig.savefig(png, dpi=140)
            print(f"[robust] plot -> {png}")
        except Exception as e:  # noqa: BLE001
            print(f"[robust] plot skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
