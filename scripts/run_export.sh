#!/usr/bin/env bash
# Waits for the extras chain to free the GPU, then exports ONNX (X1 milestone).
# nano@896 = the Jetson deploy target; small_d1@896 = the accuracy anchor.
set -u
cd /root/autodl-tmp/SoarVision || exit 1
export PYTHONPATH=scripts PYTHONUNBUFFERED=1
PY=/root/miniconda3/bin/python
say(){ echo "[$(date +%T)] $*"; }

say "waiting for extras chain to finish (GPU free)..."
while pgrep -f "[r]un_extras" >/dev/null 2>&1; do sleep 20; done
say "GPU free. Exporting ONNX."

say "export nano@896 (deploy target)"
$PY scripts/rfdetr_export.py --weights runs/rfdetr/nano_hires896/checkpoint_best_total.pth \
  --variant nano --resolution 896 --out-dir runs/rfdetr/export_nano || say "nano export FAILED"

say "export small_d1@896 (accuracy anchor)"
$PY scripts/rfdetr_export.py --weights runs/rfdetr/small_d1_896/checkpoint_best_total.pth \
  --variant small --resolution 896 --out-dir runs/rfdetr/export_small_d1 || say "small export FAILED"

say "DONE export"
