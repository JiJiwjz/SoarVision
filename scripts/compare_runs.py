"""
Cross-stage comparison (E0 vs E1 vs E2 …).

Aggregates, for each run directory:
  * the training curve (mAP50-95 vs epoch) from results.csv;
  * final/best overall metrics from results.csv;
  * per-class AP + miss-rate + recall-by-size from eval.py's eval_<split>.json, if present.

Produces a comparison table (CSV + console) and overlaid training-curve / per-class plots,
making the P2-head gain on small_boat / buoy easy to read off.

Usage
-----
    # after training + running eval.py on each run
    python scripts/compare_runs.py --runs runs/maritime/e0_* runs/maritime/e1_* runs/maritime/e2_*
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _common import CLASS_NAMES, REPO_ROOT


def _col(df: pd.DataFrame, needle: str) -> str | None:
    """Fuzzy column lookup — results.csv headers carry spaces/suffixes that vary by version."""
    needle = needle.lower().replace(" ", "")
    for c in df.columns:
        if needle in c.lower().replace(" ", ""):
            return c
    return None


def load_run(run_dir: Path) -> dict | None:
    csv = run_dir / "results.csv"
    if not csv.exists():
        print(f"[compare] no results.csv in {run_dir}, skipping")
        return None
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]

    map_col = _col(df, "mAP50-95")
    map50_col = _col(df, "mAP50")
    epoch_col = _col(df, "epoch")
    info = {"name": run_dir.name, "df": df, "map_col": map_col, "epoch_col": epoch_col}

    best_idx = df[map_col].idxmax() if map_col else None
    info["final"] = {
        "best_epoch": int(df[epoch_col].iloc[best_idx]) if (epoch_col and best_idx is not None) else None,
        "best_mAP50_95": float(df[map_col].iloc[best_idx]) if (map_col and best_idx is not None) else None,
        "best_mAP50": float(df[map50_col].iloc[best_idx]) if (map50_col and best_idx is not None) else None,
    }

    # Optional eval.py output for per-class detail.
    eval_json = None
    for cand in (run_dir / "eval" / "eval_test.json", run_dir / "eval" / "eval_val.json"):
        if cand.exists():
            eval_json = json.loads(cand.read_text())
            break
    info["eval"] = eval_json
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs or globs")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "compare"))
    args = ap.parse_args()

    run_dirs: list[Path] = []
    for pat in args.runs:
        run_dirs += [Path(p) for p in glob.glob(pat)]
    run_dirs = [d for d in run_dirs if d.is_dir()]
    runs = [r for d in sorted(run_dirs) if (r := load_run(d))]
    if not runs:
        print("[compare] no valid runs found")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- comparison table -------------------------------------------------
    rows = []
    for r in runs:
        row = {"run": r["name"], **r["final"]}
        ev = r["eval"]
        if ev:
            o = ev.get("official", {})
            row["fps"] = o.get("fps")
            for name in CLASS_NAMES:
                pc = o.get("per_class", {}).get(name, {})
                row[f"AP50_95[{name}]"] = pc.get("AP50_95")
            cu = ev.get("custom", {})
            for name in CLASS_NAMES:
                row[f"miss%[{name}]"] = (cu.get("per_class", {}).get(name, {}).get("miss_rate") or 0) * 100
            rbs = cu.get("recall_by_size", {})
            row["recall_small%"] = (rbs.get("small") or 0) * 100
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "comparison.csv", index=False)
    print("\n=== Stage comparison ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table.to_string(index=False))

    # ---- training curves --------------------------------------------------
    plt.figure(figsize=(7, 5))
    for r in runs:
        df, mc, ec = r["df"], r["map_col"], r["epoch_col"]
        if mc and ec:
            plt.plot(df[ec], df[mc], label=r["name"])
    plt.xlabel("epoch"); plt.ylabel("mAP50-95"); plt.title("Training curves")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "training_curves.png", dpi=140)

    # ---- per-class AP bars (only runs with eval.py output) ---------------
    have_eval = [r for r in runs if r["eval"]]
    if have_eval:
        import numpy as np
        x = np.arange(len(CLASS_NAMES))
        width = 0.8 / len(have_eval)
        plt.figure(figsize=(8, 5))
        for i, r in enumerate(have_eval):
            pc = r["eval"]["official"]["per_class"]
            vals = [pc.get(n, {}).get("AP50_95", 0) for n in CLASS_NAMES]
            plt.bar(x + i * width, vals, width, label=r["name"])
        plt.xticks(x + width * (len(have_eval) - 1) / 2, CLASS_NAMES)
        plt.ylabel("AP50-95"); plt.title("Per-class AP by run")
        plt.legend(); plt.grid(alpha=0.3, axis="y")
        plt.tight_layout(); plt.savefig(out_dir / "per_class_ap.png", dpi=140)

    print(f"\n[compare] saved → {out_dir} (comparison.csv, training_curves.png, per_class_ap.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
