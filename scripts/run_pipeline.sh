#!/usr/bin/env bash
# Unattended overnight experiment queue for the rented 5090 box.
# Launch DETACHED:  setsid bash scripts/run_pipeline.sh > /root/autodl-tmp/pipeline.log 2>&1 < /dev/null &
#
# Order is value-first so partial completion still yields the key result:
#   0) wait for the currently-running training (hires nano@896) to finish
#   1) eval + curves + viz for hires nano@896   <-- the small-object verdict (CRITICAL)
#   2) train small@896 (early-stop)             <-- stronger main model; partial-OK
#   3) eval + curves + viz for small@896
# No network needed: nano + small pretrained weights must be pre-cached in
# /root/.roboflow/models/. Does NOT power off the box (release it from the panel).
set -u
cd /root/autodl-tmp/SoarVision || exit 1
PY=/root/miniconda3/bin/python
export PYTHONPATH=scripts

say() { echo "[$(date +%F_%H:%M:%S)] $*"; }
HIRES=runs/rfdetr/nano_hires896
SMALL=runs/rfdetr/small_hires896

eval_run() {  # $1 = run dir, $2 = variant
  local dir="$1" variant="$2" ck="$1/checkpoint_best_total.pth"
  if [ ! -f "$ck" ]; then say "SKIP eval — no checkpoint at $ck"; return; fi
  say "eval $variant: $ck"
  $PY scripts/rfdetr_eval.py --weights "$ck" --variant "$variant" --split test || say "eval FAILED ($variant)"
  $PY scripts/plot_curves.py --run-dir "$dir" || say "curves FAILED ($variant)"
  $PY scripts/rfdetr_viz.py --weights "$ck" --variant "$variant" --num 12 || say "viz FAILED ($variant)"
  say "eval/curves/viz done ($variant)"
}

say "PIPELINE START"

# 0) wait for the in-flight training to complete
say "waiting for in-flight rfdetr_train.py to finish ..."
while pgrep -f "[r]fdetr_train.py" >/dev/null 2>&1; do sleep 60; done
say "in-flight training finished."

# 1) hires nano@896 — the small-object verdict
say "=== STEP 1: eval hires nano@896 ==="
eval_run "$HIRES" nano

# 2) train small@896 (weights must be pre-cached; partial run is fine)
say "=== STEP 2: train small@896 ==="
$PY scripts/rfdetr_train.py --variant small --dataset-dir datasets/maritime_rfdetr_hires \
    --resolution 896 --epochs 50 --num-workers 24 --batch-size 4 --grad-accum-steps 4 \
    --early-stopping --output-dir "$SMALL" || say "small@896 train FAILED/cut"
say "small@896 training step returned."

# 3) eval small@896 (whatever best it reached)
say "=== STEP 3: eval small@896 ==="
eval_run "$SMALL" small

say "PIPELINE DONE — all queued jobs complete. Release the instance from the AutoDL panel."
