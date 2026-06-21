#!/usr/bin/env bash
# Post-overnight extras: finish nano-D1 (eval+robustness from the partial G ckpt) and
# render the fog qualitative comparison (baseline vs D1). Reuses datasets/Maritime_Degraded.
set -u
cd /root/autodl-tmp/SoarVision || exit 1
export PYTHONPATH=scripts PYTHONUNBUFFERED=1
PY=/root/miniconda3/bin/python
say(){ echo "[$(date +%T)] $*"; }
NANOD1=runs/rfdetr/nano_d1_896/checkpoint_best_ema.pth   # G was cut before _total was written

say "1/3 nano-D1 clean eval"
$PY scripts/rfdetr_eval.py --weights "$NANOD1" --variant nano --resolution 896 --split test \
  || say "eval FAILED"

say "2/3 nano-D1 robustness table"
$PY scripts/rfdetr_robustness.py --weights "$NANOD1" --variant nano --resolution 896 \
  --degraded-root datasets/Maritime_Degraded --plot --time-n 100 \
  --out-dir runs/rfdetr/nano_d1_896/robustness || say "robustness FAILED"

say "3/3 fog qualitative viz: GT | baseline | D1 (fog_heavy)"
$PY scripts/rfdetr_viz.py \
  --weights  runs/rfdetr/small_hires896_v2/checkpoint_best_total.pth --variant  small \
  --weights2 runs/rfdetr/small_d1_896/checkpoint_best_total.pth      --variant2 small \
  --resolution 896 --resolution2 896 --label1 baseline --label2 D1 --conf 0.3 \
  --images-dir datasets/Maritime_Degraded/fog_heavy/images \
  --labels-dir datasets/Maritime_Degraded/fog_heavy/labels \
  --num 12 --out-dir runs/rfdetr/viz_fog_compare || say "viz FAILED"

say "DONE extras"
