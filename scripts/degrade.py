"""
Maritime image degradation synthesis — fog / low-light / sensor-noise.

The P0 data foundation for the degradation-robust line (D1 online joint training +
the graded degraded test sets that feed the robustness table). Framework-agnostic:
pure numpy + PIL, operates on HWC uint8 or float [0,1]; no detector dependency, so it
plugs into RF-DETR training (online) and our eval pipeline (offline) alike.

Each corruption is grounded in the literature, not hand-waved:

* Fog — atmospheric scattering (Koschmieder) model  I(x) = J(x)·t(x) + A·(1−t(x)),
  with transmission t(x) = exp(−β·d(x)).
    - Narasimhan & Nayar, "Vision and the Atmosphere", IJCV 2002.
    - He, Sun, Tang, "Single Image Haze Removal Using Dark Channel Prior", CVPR 2009.
    - Sakaridis, Dai, Van Gool, "Semantic Foggy Scene Understanding with Synthetic
      Data", IJCV 2018 (Foggy Cityscapes) — the recipe for *adding* synthetic fog to
      real clear images via this model. They use stereo depth; SeaShips/SMD have none,
      so we use a maritime depth *proxy* (fog denser toward the horizon) — documented
      as an approximation, not ground-truth depth.

* Severity grading (5 levels) and the corruption taxonomy (fog / gaussian_noise /
  shot_noise / brightness) follow:
    - Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common
      Corruptions and Perturbations", ICLR 2019 (ImageNet-C).

* Sensor noise — heteroscedastic Poisson–Gaussian (photon shot + read noise),
  var(I) = a·I + b, applied in the linear domain:
    - Foi, Trimeche, Katkovnik, Egiazarian, "Practical Poissonian-Gaussian Noise
      Modeling and Fitting for Single-Image Raw-Data", IEEE TIP 2008.

* Low-light — reduce illumination in *linear* (un-gamma'd) space, then add
  Poisson–Gaussian noise (noise is amplified in the dark), then re-apply sRGB:
    - Brooks et al., "Unprocessing Images for Learned Raw Denoising", CVPR 2019.
    - Chen et al., "Learning to See in the Dark", CVPR 2018.
    - Wei et al., "Deep Retinex Decomposition for Low-Light Enhancement", BMVC 2018
      (Retinex illumination scaling / gamma).

Usage
-----
    python scripts/degrade.py --selftest
    python scripts/degrade.py --demo-image path/to/img.jpg --out runs/degrade_demo
    python scripts/degrade.py --gen --split test --out datasets/Maritime_Degraded
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------------
# severity presets (1..5, ImageNet-C style). Tuned for 1080p maritime frames.
# ----------------------------------------------------------------------------------
# Fog: scattering coefficient β (transmission t=exp(-β·d), depth proxy d∈[0,1]).
FOG_BETA = {1: 0.7, 2: 1.1, 3: 1.6, 4: 2.3, 5: 3.2}
# Low-light: linear-domain exposure gain (<1 darkens).
LOWLIGHT_EXPOSURE = {1: 0.55, 2: 0.40, 3: 0.27, 4: 0.16, 5: 0.09}
# Gaussian noise std on the [0,1] sRGB image (ImageNet-C gaussian_noise).
GAUSS_SIGMA = {1: 0.04, 2: 0.07, 3: 0.10, 4: 0.15, 5: 0.21}
# Poisson-Gaussian (Foi) linear-domain coefficients (a=shot, b=read).
PG_SHOT = {1: 0.004, 2: 0.008, 3: 0.014, 4: 0.024, 5: 0.040}
PG_READ = {1: 1e-4, 2: 3e-4, 3: 6e-4, 4: 1.2e-3, 5: 2.0e-3}

# The condition set used by the robustness table (see docs/evaluation-plan.md).
TEST_CONDITIONS = {
    "clear": None,
    "fog_light": ("fog", 1),
    "fog_medium": ("fog", 3),
    "fog_heavy": ("fog", 5),
    "lowlight": ("lowlight", 3),
    "noise": ("noise", 3),
}


# ----------------------------------------------------------------------------------
# sRGB <-> linear (IEC 61966-2-1). Degradation physics live in linear light.
# ----------------------------------------------------------------------------------
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * (x ** (1 / 2.4)) - 0.055)


def _to_float(img) -> tuple[np.ndarray, bool]:
    """Accept PIL or np uint8/float HWC → float32 [0,1]. Returns (arr, was_uint8)."""
    if isinstance(img, Image.Image):
        img = np.asarray(img)
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0, True
    return arr.astype(np.float32), False


def _restore(arr: np.ndarray, was_uint8: bool):
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8) if was_uint8 else arr


# ----------------------------------------------------------------------------------
# depth proxy — maritime scenes: far (horizon) is up, near is down.
# ----------------------------------------------------------------------------------
def vertical_depth_proxy(h: int, w: int, d_near: float = 0.25, d_far: float = 1.0) -> np.ndarray:
    """d∈[d_near,d_far], larger toward the top (horizon) → fog denser far away.
    A stand-in for the stereo depth Sakaridis used; honest approximation for
    horizon-bearing sea frames (NOT geometrically exact)."""
    col = np.linspace(d_far, d_near, h, dtype=np.float32)  # row 0 (top) = far
    return np.repeat(col[:, None], w, axis=1)


# ----------------------------------------------------------------------------------
# individual corruptions
# ----------------------------------------------------------------------------------
def add_fog(img, severity: int = 3, beta: float | None = None,
            airlight: float = 0.85, depth_mode: str = "vertical", rng=None):
    """Koschmieder scattering: I = J·t + A·(1−t), t = exp(−β·d). [Narasimhan&Nayar'02,
    He'09, Sakaridis'18]"""
    rng = rng or np.random.default_rng()
    arr, u8 = _to_float(img)
    h, w = arr.shape[:2]
    beta = beta if beta is not None else FOG_BETA[severity]
    if depth_mode == "uniform":
        d = np.ones((h, w), np.float32)
    else:
        d = vertical_depth_proxy(h, w)
    t = np.exp(-beta * d)[..., None]
    A = float(np.clip(airlight + rng.uniform(-0.05, 0.05), 0.5, 1.0))
    out = arr * t + A * (1.0 - t)
    return _restore(out, u8)


def add_poisson_gaussian(lin: np.ndarray, a: float, b: float, rng) -> np.ndarray:
    """Heteroscedastic sensor noise in linear light: std = sqrt(a·I + b). [Foi'08]"""
    var = a * np.clip(lin, 0.0, None) + b
    return lin + rng.standard_normal(lin.shape).astype(np.float32) * np.sqrt(var)


def lower_light(img, severity: int = 3, exposure: float | None = None, rng=None):
    """Darken in linear domain + amplify-in-dark noise. [Brooks'19, Chen'18, Wei'18]"""
    rng = rng or np.random.default_rng()
    arr, u8 = _to_float(img)
    exposure = exposure if exposure is not None else LOWLIGHT_EXPOSURE[severity]
    lin = srgb_to_linear(arr) * exposure
    lin = add_poisson_gaussian(lin, PG_SHOT[severity], PG_READ[severity], rng)
    return _restore(linear_to_srgb(lin), u8)


def add_gaussian_noise(img, severity: int = 3, sigma: float | None = None, rng=None):
    """Additive Gaussian on the sRGB image (ImageNet-C gaussian_noise). [Hendrycks'19]"""
    rng = rng or np.random.default_rng()
    arr, u8 = _to_float(img)
    sigma = sigma if sigma is not None else GAUSS_SIGMA[severity]
    return _restore(arr + rng.standard_normal(arr.shape).astype(np.float32) * sigma, u8)


def add_sensor_noise(img, severity: int = 3, rng=None):
    """Poisson-Gaussian sensor noise (linear domain), the physical alternative to the
    plain sRGB Gaussian above. [Foi'08]"""
    rng = rng or np.random.default_rng()
    arr, u8 = _to_float(img)
    lin = srgb_to_linear(arr)
    lin = add_poisson_gaussian(lin, PG_SHOT[severity], PG_READ[severity], rng)
    return _restore(linear_to_srgb(lin), u8)


CORRUPTIONS = {
    "fog": add_fog, "lowlight": lower_light,
    "noise": add_gaussian_noise, "sensor_noise": add_sensor_noise,
}


def apply(img, kind: str, severity: int = 3, rng=None):
    return CORRUPTIONS[kind](img, severity=severity, rng=rng)


# ----------------------------------------------------------------------------------
# online transform for D1 joint degradation training
# ----------------------------------------------------------------------------------
class OnlineDegrade:
    """Per-image stochastic degradation for joint clear+degraded training (D1).

    Each corruption fires independently with its probability at a random severity in
    `severity`. Keep p's modest so a healthy fraction of clear images remain — D1's
    point is JOINT clear+degraded, not all-degraded. Geometry is untouched, so YOLO
    boxes stay valid. Plug into any dataloader: out = OnlineDegrade()(img)."""

    def __init__(self, p_fog=0.30, p_lowlight=0.20, p_noise=0.20,
                 severity=(1, 4), order=("fog", "lowlight", "noise"), seed=None):
        self.p = {"fog": p_fog, "lowlight": p_lowlight, "noise": p_noise}
        self.severity = severity
        self.order = order
        self.rng = np.random.default_rng(seed)

    def __call__(self, img):
        was_pil = isinstance(img, Image.Image)
        out = img
        for kind in self.order:
            if self.rng.random() < self.p[kind]:
                s = int(self.rng.integers(self.severity[0], self.severity[1] + 1))
                out = CORRUPTIONS[kind](out, severity=s, rng=self.rng)
        return Image.fromarray(out) if was_pil and not isinstance(out, Image.Image) else out


# ----------------------------------------------------------------------------------
# offline graded test-set generation
# ----------------------------------------------------------------------------------
def generate_testsets(src_images: Path, src_labels: Path, out_root: Path, seed: int = 0):
    """Write one degraded copy of the split per TEST_CONDITION (clear = plain copy),
    labels copied unchanged (degradation is photometric → boxes invariant). Layout
    mirrors the dataset so rfdetr_eval.py can run per condition."""
    from _common import IMG_EXTS  # local import keeps the module standalone-importable

    rng = np.random.default_rng(seed)
    imgs = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"[gen] {len(imgs)} images × {len(TEST_CONDITIONS)} conditions -> {out_root}")
    for cond, spec in TEST_CONDITIONS.items():
        io_dir = out_root / cond / "images"
        lo_dir = out_root / cond / "labels"
        io_dir.mkdir(parents=True, exist_ok=True)
        lo_dir.mkdir(parents=True, exist_ok=True)
        for p in imgs:
            im = Image.open(p).convert("RGB")
            if spec is not None:
                kind, sev = spec
                im = Image.fromarray(CORRUPTIONS[kind](im, severity=sev, rng=rng))
            im.save(io_dir / p.name)
            lbl = src_labels / (p.stem + ".txt")
            if lbl.exists():
                (lo_dir / lbl.name).write_text(lbl.read_text())
        print(f"[gen]   {cond}: done")
    print(f"[gen] saved -> {out_root}")


# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------
def _panel(img_path: Path, out: Path):
    """Save a clear-vs-each-corruption strip for visual QA."""
    base = Image.open(img_path).convert("RGB")
    rng = np.random.default_rng(0)
    tiles = [("clear", np.asarray(base))]
    for label, (kind, sev) in [("fog s3", ("fog", 3)), ("fog s5", ("fog", 5)),
                               ("lowlight s3", ("lowlight", 3)), ("noise s3", ("noise", 3)),
                               ("sensor s3", ("sensor_noise", 3))]:
        tiles.append((label, CORRUPTIONS[kind](base, severity=sev, rng=rng)))
    from PIL import ImageDraw
    h = base.size[1]
    panel = Image.new("RGB", (base.size[0] * len(tiles) + 4 * (len(tiles) - 1), h), (15, 15, 15))
    x = 0
    for label, arr in tiles:
        t = Image.fromarray(np.asarray(arr, np.uint8))
        ImageDraw.Draw(t).text((6, 6), label, fill=(255, 60, 60))
        panel.paste(t, (x, 0))
        x += base.size[0] + 4
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{img_path.stem}_degrade.jpg"
    panel.save(dst)
    print(f"[demo] saved -> {dst}")


def _selftest() -> int:
    rng = np.random.default_rng(0)
    img = (rng.uniform(0, 255, (64, 96, 3))).astype(np.uint8)
    for kind in CORRUPTIONS:
        for s in range(1, 6):
            out = CORRUPTIONS[kind](img, severity=s, rng=rng)
            assert out.shape == img.shape and out.dtype == np.uint8, (kind, s, out.shape, out.dtype)
            assert out.min() >= 0 and out.max() <= 255
    # sRGB round-trip sanity
    x = rng.random((50, 50, 3)).astype(np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-4)
    # fog at severity 5 should brighten (airlight) and reduce contrast vs clear
    foggy = add_fog(img, severity=5, rng=rng).astype(np.float32)
    assert foggy.std() < img.astype(np.float32).std()
    # OnlineDegrade returns same type, valid range
    od = OnlineDegrade(p_fog=1.0, p_lowlight=1.0, p_noise=1.0, seed=1)
    o = od(Image.fromarray(img))
    assert isinstance(o, Image.Image)
    print("[selftest] OK — sRGB round-trip, 4 corruptions × 5 severities, fog↓contrast, "
          "OnlineDegrade type-preserving")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo-image", default=None, help="degrade one image into a QA strip")
    ap.add_argument("--gen", action="store_true", help="generate graded test sets over a split")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", type=Path, default=Path("runs/degrade_demo"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if args.demo_image:
        _panel(Path(args.demo_image), args.out)
        return 0
    if args.gen:
        from _common import images_dir, labels_dir
        generate_testsets(images_dir(args.split), labels_dir(args.split), args.out, args.seed)
        return 0
    ap.error("pass --selftest | --demo-image <path> | --gen")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
