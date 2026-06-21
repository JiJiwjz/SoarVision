"""
A4 · custom training loop for the radar-fusion RF-DETR.

rfdetr.train() (PyTorch-Lightning) only knows images+COCO targets — it can't feed the
REVP map. So we drive our own loop, reusing RF-DETR's own criterion/matcher (built via
build_criterion_from_config) exactly as their LightningModule.training_step does:

    outputs   = lwdetr(samples, targets)            # samples = NestedTensor
    loss_dict = criterion(outputs, targets)
    loss      = sum(loss_dict[k]*weight_dict[k] for k in loss_dict if k in weight_dict)

The only addition is `stash_radar(model, revp)` before each forward (radar/integrate).
Targets are DETR dicts {"labels": Long[N], "boxes": Float[N,4] cxcywh-normalised} — YOLO
boxes ARE cxcywh-normalised, so they drop straight in.

    python radar/train_fusion.py --smoke              # synthetic, proves the train step
    python radar/train_fusion.py --root datasets/WaterScenes --variant nano --epochs 5
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build(variant: str, num_classes: int, resolution: int, device: str):
    """Return (rfdetr_wrapper, lwdetr_module, criterion) with radar fusion attached."""
    import rfdetr
    from rfdetr.models.lwdetr import build_criterion_from_config

    import integrate  # radar/integrate.py

    cls = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}[variant]
    m = getattr(rfdetr, cls)(num_classes=num_classes, resolution=resolution)
    # TrainConfig requires dataset_dir (a required field); criterion ignores it.
    train_config = m.get_train_config(dataset_dir=".")
    criterion, _ = build_criterion_from_config(m.model_config, train_config)
    integrate.attach_radar_fusion(m)
    lwdetr = m.model.model
    lwdetr.to(device).train()
    criterion.to(device)
    return m, lwdetr, criterion


def weighted_loss(loss_dict, weight_dict):
    return sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)


def preprocess_images(imgs_uint8: list[np.ndarray], res: int, device: str):
    """list of HWC uint8 -> NestedTensor of normalised [3,res,res] (square resize)."""
    from rfdetr.util.misc import nested_tensor_from_tensor_list

    ts = []
    for im in imgs_uint8:
        t = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        t = nn.functional.interpolate(t[None], size=(res, res), mode="bilinear",
                                      align_corners=False)[0]
        ts.append(t)
    batch = torch.stack(ts)
    batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
    samples = nested_tensor_from_tensor_list(list(batch))
    return samples.to(device)


def make_targets(boxes_list: list[np.ndarray], device: str):
    """list of [N,5] (cls,cx,cy,w,h norm) -> DETR target dicts."""
    out = []
    for b in boxes_list:
        if len(b):
            out.append({"labels": torch.as_tensor(b[:, 0], dtype=torch.long, device=device),
                        "boxes": torch.as_tensor(b[:, 1:5], dtype=torch.float, device=device)})
        else:
            out.append({"labels": torch.zeros(0, dtype=torch.long, device=device),
                        "boxes": torch.zeros(0, 4, dtype=torch.float, device=device)})
    return out


def _smoke(variant: str, device: str) -> int:
    res = 896
    m, lwdetr, criterion = build(variant, num_classes=3, resolution=res, device=device)
    import integrate

    # synthetic batch: 2 images, a few boxes each, a REVP map
    B = 2
    imgs = [(np.random.rand(540, 960, 3) * 255).astype(np.uint8) for _ in range(B)]
    boxes = [np.array([[0, 0.5, 0.5, 0.2, 0.3], [2, 0.3, 0.4, 0.05, 0.05]], np.float32),
             np.array([[1, 0.6, 0.6, 0.1, 0.1]], np.float32)]
    samples = preprocess_images(imgs, res, device)
    targets = make_targets(boxes, device)
    revp = torch.rand(B, 5, res // 8, res // 8, device=device)
    revp[:, 4] = (revp[:, 4] > 0.7).float()

    integrate.stash_radar(m, revp)
    outputs = lwdetr(samples, targets)
    loss_dict = criterion(outputs, targets)
    loss = weighted_loss(loss_dict, criterion.weight_dict)
    print(f"[smoke] forward OK; loss={loss.item():.4f}  keys={sorted(k for k in loss_dict if k in criterion.weight_dict)[:6]}...")
    assert torch.isfinite(loss), "loss not finite"

    loss.backward()
    g = [(p.grad.norm().item()) for p in integrate.radar_parameters(m) if p.grad is not None]
    print(f"[smoke] backward OK; {len(g)} radar params got grad, mean|grad|={np.mean(g):.2e}")
    assert g and np.mean(g) > 0, "no gradient reached the radar fusion params"
    print("[smoke] OK — radar-fusion train step (forward+loss+backward) works end-to-end.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="synthetic 1-step training smoke test")
    ap.add_argument("--variant", default="nano", choices=["nano", "small", "medium"])
    ap.add_argument("--root", default="datasets/WaterScenes")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.smoke:
        return _smoke(args.variant, args.device)
    raise SystemExit("real-data training loop: TODO (wire radar/dataset.py loader); smoke first")


if __name__ == "__main__":
    raise SystemExit(main())
