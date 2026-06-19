"""
Train YOLO26n on Maritime_Detection_YOLO (vessel / small_boat / buoy).

Stages (see docs/yolo26-demo-plan.md §1.1):
    e0  yolo26n.pt baseline                                       — pipeline + floor
    e1  yolo26-p2.yaml + load yolo26n.pt                          — add P2 small-object head
    e2  configs/yolo26n-p2p4.yaml + load yolo26n.pt               — also drop P5

Examples
--------
    # Smoke test: 20% subset, 30 epochs
    python scripts/train.py --stage e0 --smoke

    # Full E0 baseline
    python scripts/train.py --stage e0

    # E1 with online degradation (D1)
    python scripts/train.py --stage e1 --d1

    # E2 custom yaml with explicit overrides
    python scripts/train.py --stage e2 --epochs 150 --batch 8 --imgsz 640
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "datasets" / "Maritime_Detection_YOLO" / "data.yaml"
E2_YAML = REPO_ROOT / "configs" / "yolo26n-p2p4.yaml"

# 8GB-tuned defaults from docs/yolo26-demo-plan.md §3.2.
STAGE_DEFAULTS = {
    "e0": {
        "model_yaml": None,                 # use pretrained .pt directly
        "weights": "yolo26n.pt",
        "batch": 16,
        "name": "e0_baseline_26n",
    },
    "e1": {
        "model_yaml": "yolo26-p2.yaml",     # ships with ultralytics
        "weights": "yolo26n.pt",
        "batch": 8,
        "name": "e1_p2_26n",
    },
    "e2": {
        "model_yaml": str(E2_YAML),         # our custom drop-P5 head
        "weights": "yolo26n.pt",
        "batch": 8,
        "name": "e2_p2p4_26n",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=list(STAGE_DEFAULTS), required=True)
    p.add_argument("--data", default=str(DEFAULT_DATA), help="dataset yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=960,
                   help="960 = sweet spot for 1080p small targets; use 640 for smoke/baseline")
    p.add_argument("--batch", type=int, default=None, help="override stage default")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--cache", default="disk", help="disk | ram | False")
    p.add_argument("--copy-paste", type=float, default=0.3, help="long-tail relief for buoy/small_boat")
    p.add_argument("--project", default=str(REPO_ROOT / "runs" / "maritime"))
    p.add_argument("--name", default=None, help="override stage default run name")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="20%% fraction + 30 epochs for fast pipeline check (~<1.5h on 4060)")
    p.add_argument("--d1", action="store_true",
                   help="enable D1 online degradation (fog/low-light/noise) — see scripts/d1_degrade.py")
    p.add_argument("--fraction", type=float, default=None,
                   help="override smoke fraction; default 1.0 normally / 0.2 with --smoke")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stage = STAGE_DEFAULTS[args.stage]

    # D1 must be installed BEFORE the model/trainer instantiates its dataloader.
    if args.d1:
        from d1_degrade import enable_d1
        enable_d1()

    from ultralytics import YOLO

    if stage["model_yaml"]:
        model = YOLO(stage["model_yaml"]).load(stage["weights"])
    else:
        model = YOLO(stage["weights"])

    epochs = 30 if args.smoke else args.epochs
    fraction = args.fraction if args.fraction is not None else (0.2 if args.smoke else 1.0)
    batch = args.batch if args.batch is not None else stage["batch"]
    name = args.name or (stage["name"] + ("_smoke" if args.smoke else "") + ("_d1" if args.d1 else ""))

    # `cache` accepts string ("disk"/"ram") or bool — translate "False"/"0".
    cache = args.cache
    if isinstance(cache, str) and cache.lower() in {"false", "0", "no"}:
        cache = False

    print(f"[train] stage={args.stage} weights={stage['weights']} yaml={stage['model_yaml']}")
    print(f"[train] epochs={epochs} batch={batch} imgsz={args.imgsz} fraction={fraction} d1={args.d1}")
    print(f"[train] data={args.data}")

    model.train(
        data=args.data,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        amp=True,
        cache=cache,
        patience=args.patience,
        copy_paste=args.copy_paste,
        fraction=fraction,
        project=args.project,
        name=name,
        resume=args.resume,
        seed=args.seed,
        exist_ok=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
