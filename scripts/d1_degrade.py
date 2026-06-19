"""
D1 online degradation augmentation (fog / low-light / Gaussian noise).

Monkey-patches `ultralytics.data.augment.Albumentations` so the existing
Albumentations Compose grows one extra `OneOf([RandomFog, RandomBrightnessContrast,
GaussNoise], p=0.5)` block. Image-only ops — bboxes are untouched.

Why monkey-patch instead of editing site-packages: pip upgrades wipe the source
edit. Doing it at runtime keeps the change in our repo and survives upgrades,
at the cost of being sensitive to Ultralytics' internal class shape.

Call `enable_d1()` BEFORE constructing the YOLO trainer. `disable_d1()` reverts.
"""

from __future__ import annotations

_ORIG_INIT = None
_PATCHED = False


def _build_degrade_oneof():
    """Mirror docs/yolo26-demo-plan.md §3.4 — fog / low-light / noise, p=0.5."""
    import albumentations as A

    # Albumentations renamed several kwargs between 1.3 and 1.4+. Try the modern
    # signature first, then fall back to the legacy one used in the plan.
    try:
        fog = A.RandomFog(fog_coef_range=(0.1, 0.5), p=1.0)
    except TypeError:
        fog = A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, p=1.0)

    try:
        noise = A.GaussNoise(std_range=(0.04, 0.20), p=1.0)
    except TypeError:
        noise = A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)

    low_light = A.RandomBrightnessContrast(
        brightness_limit=(-0.5, -0.1),
        contrast_limit=(-0.3, 0.0),
        p=1.0,
    )
    return A.OneOf([fog, low_light, noise], p=0.5)


def enable_d1():
    """Install the D1 degrade transform into Ultralytics' Albumentations pipeline."""
    global _ORIG_INIT, _PATCHED
    if _PATCHED:
        return

    import albumentations as A
    from ultralytics.data import augment as ua

    _ORIG_INIT = ua.Albumentations.__init__

    def patched_init(self, p=1.0):
        _ORIG_INIT(self, p=p)
        if getattr(self, "transform", None) is None:
            return  # albumentations not installed inside ultralytics' init path

        degrade = _build_degrade_oneof()

        # Preserve the original bbox params so YOLO labels keep round-tripping.
        bbox_params = None
        processors = getattr(self.transform, "processors", None)
        if processors and "bboxes" in processors:
            bbox_params = processors["bboxes"].params
        if bbox_params is None:
            bbox_params = A.BboxParams(format="yolo", label_fields=["class_labels"])

        self.transform = A.Compose(
            list(self.transform.transforms) + [degrade],
            bbox_params=bbox_params,
        )

    ua.Albumentations.__init__ = patched_init
    _PATCHED = True
    print("[D1] Albumentations pipeline patched: +OneOf(fog/low-light/noise, p=0.5)")


def disable_d1():
    """Restore the original Albumentations.__init__. Mostly for tests."""
    global _ORIG_INIT, _PATCHED
    if not _PATCHED:
        return
    from ultralytics.data import augment as ua

    ua.Albumentations.__init__ = _ORIG_INIT
    _ORIG_INIT = None
    _PATCHED = False
