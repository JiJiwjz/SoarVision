#!/usr/bin/env bash
# Continuation job: wait for any running eval, then train+eval medium@896
# (the variant the overnight queue never reached). Launch DETACHED.
set -u
cd /root/autodl-tmp/SoarVision || exit 1
PY=/root/miniconda3/bin/python
export PYTHONPATH=scripts
say() { echo "[$(date +%F_%H:%M:%S)] $*"; }
MED=runs/rfdetr/medium_hires896

say "CONT START — waiting for any running rfdetr_eval to finish ..."
while pgrep -f "[r]fdetr_eval.py" >/dev/null 2>&1; do sleep 30; done
say "GPU free."

say "=== train medium@896 ==="
$PY scripts/rfdetr_train.py --variant medium --dataset-dir datasets/maritime_rfdetr_hires \
    --resolution 896 --epochs 50 --num-workers 24 --batch-size 2 --grad-accum-steps 8 \
    --early-stopping --output-dir "$MED" || say "medium train FAILED/cut"
say "medium training returned."

say "=== eval medium@896 (at 896) ==="
$PY scripts/rfdetr_eval.py --weights "$MED/checkpoint_best_total.pth" \
    --variant medium --resolution 896 --split test || say "medium eval FAILED"
say "CONT DONE — medium@896 trained + evaluated."
