"""
Fine-tune RF-DETR on the maritime dataset (vessel / small_boat / buoy).

RF-DETR is the main detector line (YOLO26n stays as the lightweight baseline).
It ingests the YOLO dataset produced by scripts/make_rfdetr_dataset.py directly
(format auto-detected). COCO mAP is computed internally regardless of input format.

Prereq (on the training box — RF-DETR is NOT an ultralytics dep):
    python scripts/make_rfdetr_dataset.py        # build datasets/maritime_rfdetr
    pip install rfdetr                            # + torch from the CUDA index first

Examples
--------
    # Smoke test: small variant, few epochs
    python scripts/rfdetr_train.py --variant small --epochs 10 --smoke

    # Main run, higher resolution for distant small craft / buoys
    python scripts/rfdetr_train.py --variant medium --epochs 60 --resolution 728

    # Lightest variant for the Jetson real-time story
    python scripts/rfdetr_train.py --variant nano --epochs 60

NOTE: kwargs verified against rfdetr 1.8.0's TrainConfig (lr/batch_size/
grad_accum_steps/epochs/num_workers/output_dir/early_stopping/dataset_file).
dataset_file MUST be "yolo" — RF-DETR defaults to COCO/roboflow. `resolution`
is a model-constructor arg (not a train arg); RF-DETR validates its own
per-variant divisibility constraint, so we don't pre-check it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# DataLoader workers (num_workers>0) deadlock on fork when OpenCV's internal
# threadpool is live in the parent — a classic cv2 + fork hang that freezes
# training right after "Loading train_dataloader...". Disabling cv2 threads in
# the parent before any fork avoids it (keeps the default fork start method, so
# no dataset-pickling requirement). Harmless if cv2 isn't the culprit.
try:
    import cv2
    cv2.setNumThreads(0)
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "datasets" / "maritime_rfdetr"

# --variant -> rfdetr class name. nano/small/medium are the Jetson-realistic targets;
# base/large are for the high-accuracy comparison rows (trained, not necessarily deployed).
VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="small", choices=list(VARIANTS))
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    p.add_argument("--dataset-file", default="yolo", choices=["yolo", "roboflow"],
                   help="'yolo' = make_rfdetr_dataset.py output; 'roboflow' = COCO-format. "
                        "RF-DETR defaults to roboflow, so YOLO MUST be requested explicitly")
    p.add_argument("--num-workers", type=int, default=16,
                   help="dataloader workers; large-image decode is the bottleneck for small "
                        "variants — scale toward CPU core count (e.g. 32+) so the GPU isn't starved")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4,
                   help="per-step batch; effective batch = batch_size * grad_accum_steps")
    p.add_argument("--grad-accum-steps", type=int, default=4,
                   help="keep batch_size*grad_accum ~16; raise batch_size on big GPUs and lower this")
    p.add_argument("--resolution", type=int, default=None,
                   help="÷56; default = variant's native. Higher helps small targets, costs speed")
    p.add_argument("--lr", type=float, default=None, help="override default LR")
    p.add_argument("--device", default=None, help="e.g. cuda / cuda:0; default = rfdetr's choice")
    p.add_argument("--output-dir", default=None, help="default: runs/rfdetr/<variant>")
    p.add_argument("--early-stopping", action="store_true", help="enable RF-DETR early stopping")
    p.add_argument("--smoke", action="store_true", help="few epochs for a fast pipeline check")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # RF-DETR validates the per-variant resolution constraint itself (varies by
    # version/variant — square_resize_div_64 etc.), so don't pre-guess it here.
    dataset_dir = Path(args.dataset_dir)
    if not (dataset_dir / "data.yaml").exists():
        raise SystemExit(
            f"no data.yaml under {dataset_dir} — run scripts/make_rfdetr_dataset.py first"
        )

    import rfdetr  # lazy: only needed at train time, not an ultralytics dep

    model_cls = getattr(rfdetr, VARIANTS[args.variant])
    output_dir = args.output_dir or str(REPO_ROOT / "runs" / "rfdetr" / args.variant)
    epochs = 5 if args.smoke else args.epochs

    # resolution is a constructor arg; training schedule args go to .train().
    ctor_kwargs = {}
    if args.resolution is not None:
        ctor_kwargs["resolution"] = args.resolution

    train_kwargs = {
        "dataset_dir": str(dataset_dir),
        "dataset_file": args.dataset_file,   # 'yolo' — RF-DETR defaults to COCO/roboflow
        "epochs": epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "num_workers": args.num_workers,
        "output_dir": output_dir,
        "early_stopping": args.early_stopping,
    }
    if args.lr is not None:
        train_kwargs["lr"] = args.lr
    if args.device is not None:
        train_kwargs["device"] = args.device

    print(f"[rfdetr] variant={args.variant} ({VARIANTS[args.variant]}) ctor={ctor_kwargs}")
    print(f"[rfdetr] epochs={epochs} batch={args.batch_size} grad_accum={args.grad_accum_steps} "
          f"(eff batch ≈ {args.batch_size * args.grad_accum_steps})")
    print(f"[rfdetr] dataset={dataset_dir}")
    print(f"[rfdetr] output={output_dir}")

    model = model_cls(**ctor_kwargs)
    model.train(**train_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
