"""
Inference for the YOLO26 maritime detector.

Source kinds (auto-detected by Ultralytics):
    * single image          --source path/to/img.jpg
    * folder of images      --source path/to/dir
    * glob                  --source 'path/to/dir/*.jpg'
    * video file            --source path/to/clip.mp4
    * webcam                --source 0
    * test split shortcut   --source test   (resolves to datasets/.../images/test)

Examples
--------
    python scripts/infer.py --weights runs/maritime/e1_p2_26n/weights/best.pt \
                            --source datasets/Maritime_Detection_YOLO/images/test \
                            --save-dir runs/predict/e1_test

    python scripts/infer.py --weights best.pt --source 0 --show   # live webcam
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES = REPO_ROOT / "datasets" / "Maritime_Detection_YOLO" / "images" / "test"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="path to trained .pt")
    p.add_argument("--source", required=True,
                   help="image / folder / glob / video / webcam idx; 'test' = test split")
    p.add_argument("--imgsz", type=int, default=960, help="match the training imgsz")
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--save-dir", default=str(REPO_ROOT / "runs" / "predict"))
    p.add_argument("--name", default="exp")
    p.add_argument("--show", action="store_true", help="live window (video/webcam)")
    p.add_argument("--save-txt", action="store_true", help="also write YOLO-format labels")
    p.add_argument("--save-conf", action="store_true", help="include conf in --save-txt output")
    p.add_argument("--vid-stride", type=int, default=1, help="sample every Nth video frame")
    return p.parse_args()


def resolve_source(src: str) -> str:
    if src == "test":
        return str(TEST_IMAGES)
    # Webcam index passed as int-like string.
    if src.isdigit():
        return src
    return src


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    source = resolve_source(args.source)
    # int-like webcam source must be passed as int to predict()
    source_arg = int(source) if isinstance(source, str) and source.isdigit() else source

    print(f"[infer] weights={args.weights} source={source_arg} imgsz={args.imgsz} conf={args.conf}")

    model.predict(
        source=source_arg,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show=args.show,
        vid_stride=args.vid_stride,
        project=args.save_dir,
        name=args.name,
        exist_ok=True,
        stream=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
