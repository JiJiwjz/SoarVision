#!/usr/bin/env bash
# Unattended overnight experiment queue for the RF-DETR degradation-robust line.
# Ordered by value-early: baseline robustness first (no training), then the D1 gain,
# then fillers. Each step is independently guarded — one failure does NOT stop the
# queue. Launch DETACHED:
#
#   cd /root/autodl-tmp/SoarVision
#   setsid bash scripts/run_overnight.sh > /root/autodl-tmp/overnight.boot.log 2>&1 < /dev/null &
#
# Then just watch runs/overnight_*/master.log
set -u
export PYTHONUNBUFFERED=1

ROOT=/root/autodl-tmp/SoarVision
PY=/root/miniconda3/bin/python
export PYTHONPATH=scripts
cd "$ROOT" || { echo "no $ROOT"; exit 1; }

# ---- config: datasets / checkpoints / output ----
HIRES=datasets/maritime_rfdetr_hires           # clean hi-res RF-DETR dataset
D1DS=datasets/maritime_rfdetr_hires_d1          # D1 joint clear+degraded train set (built below)
DEG=datasets/Maritime_Degraded                  # graded degraded TEST sets (built in step A, reused)
SMALL=runs/rfdetr/small_hires896_v2/checkpoint_best_total.pth   # current best (anchor)
NANO=runs/rfdetr/nano_hires896/checkpoint_best_total.pth
D1OUT=runs/rfdetr/small_d1_896
MEDOUT=runs/rfdetr/medium_hires896
RES=896

LOGDIR=$ROOT/runs/overnight_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
MASTER=$LOGDIR/master.log
say(){ echo "[$(date +%F_%T)] $*" | tee -a "$MASTER"; }

step(){ # $1=name  $2=logfile  rest=command
  local name=$1 log=$2; shift 2
  say ">> START $name"
  "$@" >"$LOGDIR/$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then say "== DONE  $name"; else say "!! FAILED $name (rc=$rc) - continuing"; fi
  echo "----- tail $log -----" | tee -a "$MASTER"
  tail -n 8 "$LOGDIR/$log" 2>/dev/null | sed 's/^/    /' | tee -a "$MASTER"
}

# ---- GPU guard: this queue needs CUDA; don't waste a night crawling on CPU ----
if ! nvidia-smi -L >/dev/null 2>&1; then
  say "!! NO GPU detected (wukamoshi / no-GPU mode?). This queue needs CUDA - aborting."
  exit 2
fi
say "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
say "logs -> $LOGDIR"

# ============================ QUEUE (value-ordered) ============================

# A · baseline robustness on the current best (small@896). --gen ALSO builds the
#     graded degraded TEST sets at $DEG, reused by every later robustness step.
step "A_baseline_small_robustness" A_baseline_small_robustness.log \
  $PY scripts/rfdetr_robustness.py --weights "$SMALL" --variant small --resolution $RES \
      --gen --split test --degraded-root "$DEG" --plot --time-n 100 \
      --out-dir runs/rfdetr/small_hires896_v2/robustness

# A2 · nano@896 robustness (model already exists; reuse $DEG, no --gen) — variant row
step "A2_nano_robustness" A2_nano_robustness.log \
  $PY scripts/rfdetr_robustness.py --weights "$NANO" --variant nano --resolution $RES \
      --degraded-root "$DEG" --plot --time-n 100 \
      --out-dir runs/rfdetr/nano_hires896/robustness

# B · build the D1 joint clear+degraded training set (offline, no RF-DETR internals)
step "B_make_d1_dataset" B_make_d1_dataset.log \
  $PY scripts/make_d1_dataset.py --src "$HIRES" --out "$D1DS" --frac 0.5

# C · train small@896 + D1 (the degradation-robust model)
step "C_train_small_d1" C_train_small_d1.log \
  $PY scripts/rfdetr_train.py --variant small --dataset-dir "$D1DS" --resolution $RES \
      --epochs 50 --early-stopping --num-workers 24 --batch-size 4 --grad-accum-steps 4 \
      --output-dir "$D1OUT"

# D · D1 clean-test eval (does D1 cost anything on clear? — the honesty check)
step "D_eval_d1_clean" D_eval_d1_clean.log \
  $PY scripts/rfdetr_eval.py --weights "$D1OUT/checkpoint_best_total.pth" --variant small \
      --resolution $RES --split test

# E · D1 robustness table (reuse $DEG) — the headline: D1 vs baseline on degraded
step "E_d1_robustness" E_d1_robustness.log \
  $PY scripts/rfdetr_robustness.py --weights "$D1OUT/checkpoint_best_total.pth" --variant small \
      --resolution $RES --degraded-root "$DEG" --plot --time-n 100 \
      --out-dir "$D1OUT/robustness"

# F · FILLER (only if the night still has time): medium@896 variant-table completion
step "F1_train_medium" F1_train_medium.log \
  $PY scripts/rfdetr_train.py --variant medium --dataset-dir "$HIRES" --resolution $RES \
      --epochs 50 --early-stopping --num-workers 24 --batch-size 2 --grad-accum-steps 8 \
      --output-dir "$MEDOUT"
step "F2_eval_medium" F2_eval_medium.log \
  $PY scripts/rfdetr_eval.py --weights "$MEDOUT/checkpoint_best_total.pth" --variant medium \
      --resolution $RES --split test
step "F3_medium_robustness" F3_medium_robustness.log \
  $PY scripts/rfdetr_robustness.py --weights "$MEDOUT/checkpoint_best_total.pth" --variant medium \
      --resolution $RES --degraded-root "$DEG" --plot --time-n 100 \
      --out-dir "$MEDOUT/robustness"

say "ALL QUEUE STEPS ATTEMPTED. Robustness tables:"
find runs -path '*/robustness/robustness_table.csv' 2>/dev/null | sed 's/^/    /' | tee -a "$MASTER"
say "Queue finished. Master log: $MASTER"
