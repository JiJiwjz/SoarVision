"""
Alignment sanity check — overlay radar points on the image + render the REVP panel.

BEFORE any fusion training, confirm the radar (u, v) actually lands on the objects in
the image. If the dots sit on ships/buoys, the WaterScenes projection is trusted; if
they are shifted, the calibration/columns are wrong and fusion would learn garbage.

Two PNGs per frame (PIL only — no matplotlib, so it runs anywhere):
  <out>/<frame>_overlay.png : image + GT boxes + radar dots (color=range, size=power)
  <out>/<frame>_revp.png    : the 5 REVP channels tiled as gray images

Usage
-----
    python radar/visualize.py --root datasets/WaterScenes --frame 000123 --out runs/radar_viz
    python radar/visualize.py --root datasets/WaterScenes --split train --num 8 --out runs/radar_viz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from revp import REVP_CHANNELS, RevpNorm, build_from_points
from waterscenes import SOAR_CLASSES, list_frames, load_classes, load_frame

# vessel / small_boat / buoy box colors
BOX_COLORS = [(0, 200, 255), (255, 180, 0), (0, 255, 120)]


def _range_to_rgb(t: float) -> tuple[int, int, int]:
    """Near (t=0) red → mid green → far (t=1) blue. Simple 3-stop ramp."""
    t = float(np.clip(t, 0, 1))
    if t < 0.5:
        s = t / 0.5
        return (int(255 * (1 - s)), int(255 * s), 0)
    s = (t - 0.5) / 0.5
    return (0, int(255 * (1 - s)), int(255 * s))


def draw_overlay(frame, norm: RevpNorm, point_radius: int = 4) -> Image.Image:
    img = Image.open(frame.image_path).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)

    # GT boxes (remapped to our 3 classes)
    for b in frame.boxes:
        cls, cx, cy, bw, bh = b
        x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
        x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
        color = BOX_COLORS[int(cls) % len(BOX_COLORS)]
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        d.text((x1 + 2, y1 + 2), SOAR_CLASSES[int(cls)], fill=color)

    # radar points: color by range, radius by normalised power
    r = frame.radar
    if len(r):
        pw_lo, pw_hi = (float(r.power.min()), float(r.power.max())) if r.power.size else (0, 1)
        for i in range(len(r)):
            u, v = float(r.u[i]), float(r.v[i])
            if not (0 <= u < W and 0 <= v < H):
                continue
            t = float(np.clip(r.rng[i] / norm.range_max, 0, 1))
            pw = (r.power[i] - pw_lo) / (pw_hi - pw_lo + 1e-6)
            rad = point_radius + int(3 * pw)
            d.ellipse([u - rad, v - rad, u + rad, v + rad],
                      outline=_range_to_rgb(t), width=2)
    return img


def draw_revp_panel(revp: np.ndarray, tile_w: int = 320) -> Image.Image:
    """Tile the C channels horizontally as gray images with labels."""
    C, H, W = revp.shape
    th = max(1, int(tile_w * H / W))
    tiles = []
    for c in range(C):
        ch = revp[c]
        g = (np.clip(ch, 0, 1) * 255).astype(np.uint8)
        t = Image.fromarray(g, mode="L").convert("RGB").resize((tile_w, th), Image.NEAREST)
        ImageDraw.Draw(t).text((4, 4), REVP_CHANNELS[c], fill=(255, 60, 60))
        tiles.append(t)
    panel = Image.new("RGB", (tile_w * C + 2 * (C - 1), th), (20, 20, 20))
    x = 0
    for t in tiles:
        panel.paste(t, (x, 0))
        x += tile_w + 2
    return panel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--frame", default=None, help="single frame id")
    ap.add_argument("--split", default=None, help="sample evenly from this split")
    ap.add_argument("--num", type=int, default=6, help="#frames when using --split")
    ap.add_argument("--downsample", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("runs/radar_viz"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    norm = RevpNorm()
    ws_classes = load_classes(args.root)

    if args.frame:
        ids = [args.frame]
    else:
        allids = list_frames(args.root, args.split)
        if not allids:
            print(f"[viz] no frames under {args.root} (split={args.split})")
            return 1
        step = max(1, len(allids) // args.num)
        ids = allids[::step][: args.num]

    for fid in ids:
        fr = load_frame(args.root, fid, ws_classes)
        if not fr.image_path.exists():
            print(f"[viz] skip {fid}: image missing")
            continue
        draw_overlay(fr, norm).save(args.out / f"{fid}_overlay.png")
        W, H = Image.open(fr.image_path).size
        revp = build_from_points(fr.radar, W, H, downsample=args.downsample, norm=norm)
        draw_revp_panel(revp).save(args.out / f"{fid}_revp.png")
        print(f"[viz] {fid}: pts={len(fr.radar)} boxes={len(fr.boxes)} -> {args.out}/{fid}_*.png")
    print(f"[viz] done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
