"""
REVP map builder — turn sparse radar points into an image-aligned multi-channel map.

This is the radar representation our RF-DETR fusion branch consumes. mmWave point
clouds are sparse and have no texture, so the WaterScenes-style trick is to render
them as a low-resolution, image-aligned tensor (one cell per image patch) whose
channels carry the physical quantities the camera CANNOT see through fog:

    channel 0  range       — radial distance (m), the pseudo-depth prior
    channel 1  elevation   — elevation angle (deg)
    channel 2  doppler     — radial velocity (m/s), separates movers from clutter
    channel 3  power       — reflected power (dB), target strength / RCS proxy
    channel 4  occupancy   — 1 where any radar point landed, else 0 (presence mask)

Points are placed by their precomputed image coordinates (u, v) divided by the
downsample factor. When several points fall in one cell we keep the CLOSEST one
(min range) by default — nearest surface dominates — configurable to max-power.

The map is deliberately framework-agnostic (numpy). dataset.py wraps it for torch.

Self-test (no dataset needed)::

    python radar/revp.py --selftest
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

REVP_CHANNELS = ["range", "elevation", "doppler", "power", "occupancy"]
NUM_CHANNELS = len(REVP_CHANNELS)


@dataclass
class RevpNorm:
    """Normalisation constants → each channel roughly in [0,1] (occupancy already 0/1).

    Defaults sized for the Oculii EAGLE 77 GHz radar (200 m range). Tune once real
    histograms are seen; power uses per-frame min-max because dB offset varies."""

    range_max: float = 200.0      # m  (sensor spec)
    elev_abs: float = 30.0        # deg (4D radar elevation FOV is narrow; clip to ±)
    doppler_abs: float = 30.0     # m/s (clip; signed → mapped to [0,1] around 0.5)
    power_per_frame: bool = True  # min-max power within the frame
    power_lo: float = 0.0         # used when power_per_frame=False
    power_hi: float = 100.0


def build_revp_map(
    u: np.ndarray,
    v: np.ndarray,
    rng: np.ndarray,
    elevation: np.ndarray,
    doppler: np.ndarray,
    power: np.ndarray,
    img_w: int,
    img_h: int,
    downsample: int = 8,
    norm: RevpNorm | None = None,
    reduce: str = "min_range",
) -> np.ndarray:
    """Scatter radar points into a [NUM_CHANNELS, H, W] float32 map (H=img_h//ds).

    reduce: collision policy when >1 point per cell — "min_range" (default) or
    "max_power". Empty cells are 0 in every channel (incl. occupancy)."""
    norm = norm or RevpNorm()
    H = max(1, img_h // downsample)
    W = max(1, img_w // downsample)
    out = np.zeros((NUM_CHANNELS, H, W), np.float32)
    if len(u) == 0:
        return out

    # image coords → grid cells, keep only in-frame points
    gx = np.floor(u / downsample).astype(np.int64)
    gy = np.floor(v / downsample).astype(np.int64)
    keep = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    if not keep.any():
        return out
    gx, gy = gx[keep], gy[keep]
    rng, elevation = rng[keep], elevation[keep]
    doppler, power = doppler[keep], power[keep]

    # normalise channels
    c_range = np.clip(rng / norm.range_max, 0.0, 1.0)
    c_elev = np.clip(elevation / norm.elev_abs, -1.0, 1.0) * 0.5 + 0.5
    c_dopp = np.clip(doppler / norm.doppler_abs, -1.0, 1.0) * 0.5 + 0.5
    if norm.power_per_frame and power.size:
        lo, hi = float(power.min()), float(power.max())
    else:
        lo, hi = norm.power_lo, norm.power_hi
    c_power = np.clip((power - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    # collision priority: lower key written last so it wins → sort so winners come last
    key = rng if reduce == "min_range" else -power
    order = np.argsort(-key)  # worst first, best (smallest key) last
    gx, gy = gx[order], gy[order]
    c_range, c_elev = c_range[order], c_elev[order]
    c_dopp, c_power = c_dopp[order], c_power[order]

    flat = gy * W + gx
    out[0].reshape(-1)[flat] = c_range
    out[1].reshape(-1)[flat] = c_elev
    out[2].reshape(-1)[flat] = c_dopp
    out[3].reshape(-1)[flat] = c_power
    out[4].reshape(-1)[flat] = 1.0  # occupancy
    return out


def build_from_points(points, img_w: int, img_h: int, **kw) -> np.ndarray:
    """Convenience for a waterscenes.RadarPoints object."""
    return build_revp_map(
        points.u, points.v, points.rng, points.elevation,
        points.doppler, points.power, img_w, img_h, **kw,
    )


# --------------------------------------------------------------------------------------
def _selftest() -> int:
    rs = np.random.default_rng(0)
    n = 200
    img_w, img_h, ds = 1920, 1080, 8
    # n in-frame points + 2 out-of-frame (negative / past-right) to test clipping
    u = np.concatenate([rs.uniform(0, img_w, n), [-5.0, img_w + 50.0]])
    v = np.concatenate([rs.uniform(0, img_h, n), [10.0, 10.0]])
    rng = np.concatenate([rs.uniform(5, 200, n), [50.0, 50.0]])
    elev = np.concatenate([rs.uniform(-30, 30, n), [0.0, 0.0]])
    dopp = np.concatenate([rs.uniform(-30, 30, n), [0.0, 0.0]])
    power = np.concatenate([rs.uniform(10, 90, n), [50.0, 50.0]])

    m = build_revp_map(u, v, rng, elev, dopp, power, img_w, img_h, downsample=ds)
    H, W = img_h // ds, img_w // ds
    assert m.shape == (NUM_CHANNELS, H, W), m.shape
    occ = m[4]
    assert occ.max() == 1.0 and set(np.unique(occ)).issubset({0.0, 1.0})
    assert (m[0][occ > 0] >= 0).all() and (m[0] <= 1).all()
    assert int(occ.sum()) <= n  # out-of-frame dropped, collisions merged
    print(f"[selftest] OK  map={m.shape}  occupied_cells={int(occ.sum())}/{H*W}  "
          f"range∈[{m[0][occ>0].min():.2f},{m[0][occ>0].max():.2f}]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    ap.error("pass --selftest (or import build_revp_map)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
