"""
WaterScenes layout + frame loader — the data side of the radar-camera fusion line.

WaterScenes (arXiv 2307.06505, https://github.com/WaterScenes/WaterScenes) is a
water-surface 4D radar-camera dataset with the SAME 77 GHz band as our TI IWR6843,
and it ships radar points ALREADY projected to image coordinates (u, v columns) —
so we do NOT need extrinsic calibration to build a radar map on WaterScenes; that is
only needed later for our own captured clips.

Expected layout (per the official toolkit)::

    <root>/
      image/        <frame>.jpg            RGB, 1920x1080
      radar/        <frame>.csv            one row per radar point (columns below)
      calib/        <frame>.txt            intrinsic + radar->camera extrinsic
      detection/    yolo/ <frame>.txt      YOLO-normalised boxes (cls cx cy w h)
                    xml/  <frame>.xml
      ImageSets/    train.txt val.txt test.txt   (frame stems, one per line) [optional]

Radar CSV columns (documented order; we still prefer the header row if present)::

    timestamp, range, doppler, azimuth, elevation, power,
    x, y, z, comp_height, comp_velocity, u, v, label, instance

We remap WaterScenes' 7 classes onto our 3 SoarVision classes (vessel / small_boat /
buoy) BY NAME, dropping shore/person targets. The id->name order below is the
documented one but MUST be verified against the dataset's own classes file after
download (load_classes() will override it if a names file exists).

Usage
-----
    python radar/waterscenes.py --root datasets/WaterScenes --validate
    python radar/waterscenes.py --root datasets/WaterScenes --frame 000123 --probe
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- SoarVision target classes (must match scripts/_common.CLASS_NAMES order) ---------
SOAR_CLASSES = ["vessel", "small_boat", "buoy"]

# --- WaterScenes class id->name (DOCUMENTED order — VERIFY after download) ------------
# The paper lists: pier, buoy, sailor, ship, boat, vessel, kayak. Their YOLO label ids
# may differ; load_classes() reads an on-disk names file when available and wins.
WATERSCENES_CLASSES = ["pier", "buoy", "sailor", "ship", "boat", "vessel", "kayak"]

# --- WaterScenes name -> SoarVision index (None = drop). Configurable on purpose. -----
REMAP: dict[str, int | None] = {
    "ship": 0,      # vessel
    "vessel": 0,    # vessel
    "boat": 1,      # small_boat
    "kayak": 1,     # small_boat
    "buoy": 2,      # buoy
    "sailor": None,  # person on deck — not a target class, drop
    "pier": None,    # shore structure — drop
}

# --- radar CSV documented column order (positional fallback when no header) -----------
RADAR_COLUMNS = [
    "timestamp", "range", "doppler", "azimuth", "elevation", "power",
    "x", "y", "z", "comp_height", "comp_velocity", "u", "v", "label", "instance",
]

IMG_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class RadarPoints:
    """Per-point radar arrays, all length N (parallel arrays, image-aligned via u,v)."""

    u: np.ndarray          # image column (px)
    v: np.ndarray          # image row (px)
    rng: np.ndarray        # range (m)
    elevation: np.ndarray  # elevation angle (deg)
    doppler: np.ndarray    # radial velocity (m/s)
    power: np.ndarray      # reflected power (dB)
    x: np.ndarray          # cartesian (m)
    y: np.ndarray
    z: np.ndarray
    label: np.ndarray      # per-point semantic class id (or -1)

    def __len__(self) -> int:
        return len(self.u)

    @classmethod
    def empty(cls) -> "RadarPoints":
        z = np.zeros((0,), np.float32)
        return cls(z, z, z, z, z, z, z, z, z, z.astype(np.int64))


@dataclass
class Frame:
    """One synchronised sample: image path + radar points + remapped YOLO boxes."""

    frame_id: str
    image_path: Path
    radar: RadarPoints
    boxes: np.ndarray  # [N,5] = soar_cls, cx, cy, w, h  (YOLO-normalised, our 3 classes)


# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------
def image_path(root: Path, frame_id: str) -> Path:
    d = root / "image"
    for ext in IMG_EXTS:
        p = d / f"{frame_id}{ext}"
        if p.exists():
            return p
    return d / f"{frame_id}.jpg"  # nominal even if missing


def radar_path(root: Path, frame_id: str) -> Path:
    return root / "radar" / f"{frame_id}.csv"


def detection_path(root: Path, frame_id: str) -> Path:
    # toolkit uses detection/yolo/<frame>.txt; fall back to detection/<frame>.txt
    p = root / "detection" / "yolo" / f"{frame_id}.txt"
    return p if p.exists() else root / "detection" / f"{frame_id}.txt"


# --------------------------------------------------------------------------------------
# class names
# --------------------------------------------------------------------------------------
def load_classes(root: Path) -> list[str]:
    """Prefer an on-disk names file (classes.txt / predefined_classes.txt); else the
    documented constant. Verifies our REMAP keys actually exist."""
    for cand in ("classes.txt", "predefined_classes.txt", "detection/classes.txt"):
        f = root / cand
        if f.exists():
            names = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
            if names:
                return names
    return list(WATERSCENES_CLASSES)


# --------------------------------------------------------------------------------------
# radar csv
# --------------------------------------------------------------------------------------
def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_radar(root: Path, frame_id: str) -> RadarPoints:
    """Read radar/<frame>.csv into parallel arrays. Header-aware; positional fallback."""
    path = radar_path(root, frame_id)
    if not path.exists():
        return RadarPoints.empty()
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return RadarPoints.empty()

    header = [c.strip() for c in rows[0]]
    has_header = any(not _is_float(c) for c in header)
    cols = header if has_header else list(RADAR_COLUMNS)
    data_rows = rows[1:] if has_header else rows
    idx = {name: i for i, name in enumerate(cols)}

    if not data_rows:
        return RadarPoints.empty()
    arr = np.array([[float(x) for x in r] for r in data_rows], dtype=np.float32)

    def col(name: str, default: float = 0.0) -> np.ndarray:
        j = idx.get(name)
        if j is None or j >= arr.shape[1]:
            return np.full((len(arr),), default, np.float32)
        return arr[:, j].astype(np.float32)

    return RadarPoints(
        u=col("u"), v=col("v"), rng=col("range"), elevation=col("elevation"),
        doppler=col("doppler"), power=col("power"),
        x=col("x"), y=col("y"), z=col("z"),
        label=col("label", -1.0).astype(np.int64),
    )


# --------------------------------------------------------------------------------------
# detection labels (YOLO) + class remap
# --------------------------------------------------------------------------------------
def load_boxes(root: Path, frame_id: str, ws_classes: list[str]) -> np.ndarray:
    """Read YOLO boxes and remap WaterScenes class ids -> SoarVision ids, dropping
    unmapped (pier/sailor). Returns [N,5] = soar_cls, cx, cy, w, h (normalised)."""
    path = detection_path(root, frame_id)
    if not path.exists():
        return np.zeros((0, 5), np.float32)
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        ws_id = int(float(parts[0]))
        name = ws_classes[ws_id] if 0 <= ws_id < len(ws_classes) else None
        soar = REMAP.get(name) if name is not None else None
        if soar is None:
            continue
        cx, cy, w, h = (float(x) for x in parts[1:5])
        out.append([soar, cx, cy, w, h])
    return np.asarray(out, np.float32) if out else np.zeros((0, 5), np.float32)


# --------------------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------------------
def list_frames(root: Path, split: str | None = None) -> list[str]:
    """Frame stems for a split. Uses ImageSets/<split>.txt if present, else every image."""
    if split:
        for cand in (root / "ImageSets" / f"{split}.txt", root / f"{split}.txt"):
            if cand.exists():
                return [ln.strip() for ln in cand.read_text().splitlines() if ln.strip()]
    img_dir = root / "image"
    if not img_dir.exists():
        return []
    return sorted(p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def load_frame(root: Path, frame_id: str, ws_classes: list[str] | None = None) -> Frame:
    ws_classes = ws_classes or load_classes(root)
    return Frame(
        frame_id=frame_id,
        image_path=image_path(root, frame_id),
        radar=load_radar(root, frame_id),
        boxes=load_boxes(root, frame_id, ws_classes),
    )


# --------------------------------------------------------------------------------------
# CLI: validate a downloaded root / probe one frame
# --------------------------------------------------------------------------------------
def validate_root(root: Path) -> int:
    print(f"[ws] root = {root}")
    ok = True
    for sub in ("image", "radar", "detection"):
        d = root / sub
        present = d.exists()
        ok &= present
        print(f"  {'OK ' if present else 'MISS'} {sub}/")
    ws_classes = load_classes(root)
    print(f"  classes ({len(ws_classes)}): {ws_classes}")
    missing = [k for k in REMAP if k not in ws_classes]
    if missing:
        print(f"  ⚠ REMAP keys not in class list (verify names/order!): {missing}")
    frames = list_frames(root)
    print(f"  frames found: {len(frames)}")
    if frames:
        for split in ("train", "val", "test"):
            n = len(list_frames(root, split))
            print(f"    split {split}: {n}")
        fr = load_frame(root, frames[0], ws_classes)
        print(f"  probe {fr.frame_id}: img={fr.image_path.name} "
              f"radar_pts={len(fr.radar)} boxes={len(fr.boxes)}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--frame", default=None, help="frame id to probe")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    if args.validate:
        return validate_root(args.root)
    if args.frame and args.probe:
        fr = load_frame(args.root, args.frame)
        print(f"frame {fr.frame_id}")
        print(f"  image: {fr.image_path}  (exists={fr.image_path.exists()})")
        r = fr.radar
        print(f"  radar points: {len(r)}")
        if len(r):
            print(f"    range  [m]   min/max = {r.rng.min():.1f}/{r.rng.max():.1f}")
            print(f"    doppler[m/s] min/max = {r.doppler.min():.1f}/{r.doppler.max():.1f}")
            print(f"    power  [dB]  min/max = {r.power.min():.1f}/{r.power.max():.1f}")
            print(f"    u in [{r.u.min():.0f},{r.u.max():.0f}] v in [{r.v.min():.0f},{r.v.max():.0f}]")
        print(f"  boxes (remapped): {len(fr.boxes)}")
        for b in fr.boxes[:10]:
            print(f"    cls={SOAR_CLASSES[int(b[0])]:<10} cxcywh={b[1:].round(3).tolist()}")
        return 0
    ap.error("pass --validate, or --frame <id> --probe")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
