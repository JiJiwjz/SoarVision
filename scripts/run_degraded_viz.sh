#!/usr/bin/env bash
# Render the latest model's inference on each degraded condition: GT | baseline | D1,
# on the SAME sampled scenes across fog/lowlight/noise (even-sampling over identical
# filenames => same frames per condition). For the 答辩 "detect-through-degradation" story.
set -u
cd /root/autodl-tmp/SoarVision || exit 1
export PYTHONPATH=scripts PYTHONUNBUFFERED=1
PY=/root/miniconda3/bin/python
say(){ echo "[$(date +%T)] $*"; }

say "waiting for extras/export chains to free the GPU..."
while pgrep -f "[r]un_extras|[r]un_export" >/dev/null 2>&1; do sleep 20; done
say "GPU free. Rendering degraded inference (GT | baseline | D1)."

BASE=runs/rfdetr/small_hires896_v2/checkpoint_best_total.pth
D1=runs/rfdetr/small_d1_896/checkpoint_best_total.pth
for cond in fog_light fog_medium fog_heavy lowlight noise; do
  say "viz $cond"
  $PY scripts/rfdetr_viz.py \
    --weights  "$BASE" --variant small \
    --weights2 "$D1"   --variant2 small \
    --resolution 896 --resolution2 896 --label1 baseline --label2 D1 --conf 0.3 \
    --images-dir "datasets/Maritime_Degraded/$cond/images" \
    --labels-dir "datasets/Maritime_Degraded/$cond/labels" \
    --num 6 --out-dir "runs/rfdetr/viz_degraded/$cond" || say "$cond FAILED"
done
say "DONE degraded viz -> runs/rfdetr/viz_degraded/<condition>/"
