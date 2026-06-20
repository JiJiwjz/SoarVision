#!/usr/bin/env bash
# Unattended overnight experiment queue for the RF-DETR degradation-robust line.
# VALUE-ORDERED: the headline results (baseline robustness + small D1 gain) run first
# and are guaranteed within budget; nano-D1 (deploy story) and medium fill the tail.
# Each step is independently guarded — one failure does NOT stop the queue.
# Budget assumed ~9h. Launch DETACHED:
#
#   cd /root/autodl-tmp/SoarVision
#   setsid bash scripts/run_overnight.sh > /root/autodl-tmp/overnight.boot.log 2>&1 < /dev/null &
#
# Watch:  tail -f runs/overnight_*/master.log   (per-STEP) ;  the per-step *.log for live progress.
set -u
export PYTHONUNBUFFERED=1

ROOT=/root/autodl-tmp/SoarVision
PY=/root/miniconda3/bin/python
export PYTHONPATH=scripts
cd "$ROOT" || { echo "no $ROOT"; exit 1; }

# ---- config: datasets / checkpoints / output ----
HIRES=datasets/maritime_rfdetr_hires            # clean hi-res RF-DETR dataset
D1DS=datasets/maritime_rfdetr_hires_d1          # D1 joint clear+degraded train set (built in B)
DEG=datasets/Maritime_Degraded                  # graded degraded TEST sets (built in A, reused)
SMALL=runs/rfdetr/small_hires896_v2/checkpoint_best_total.pth   # current best (anchor)
NANO=runs/rfdetr/nano_hires896/checkpoint_best_total.pth
SD1=runs/rfdetr/small_d1_896                     # small + D1 output
ND1=runs/rfdetr/nano_d1_896                      # nano + D1 output
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

# ---- GPU guard: this queue needs CUDA; don't waste budget crawling on CPU ----
if ! nvidia-smi -L >/dev/null 2>&1; then
  say "!! NO GPU detected (no-GPU mode?). This queue needs CUDA - aborting."
  exit 2
fi
say "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
say "logs -> $LOGDIR  | budget ~9h, value-ordered (headlines first)"

# ============================ QUEUE (value-ordered) ============================

# A · baseline robustness on the current best (small@896). --gen ALSO builds the
#     graded degraded TEST sets at $DEG, reused by every later robustness step.
step "A_baseline_small_robustness" A_baseline_small_robustness.log \
  $PY scripts/rfdetr_robustness.py --weights "$SMALL" --variant small --resolution $RES \
      --gen --split test --degraded-root "$DEG" --plot --time-n 100 \
      --out-dir runs/rfdetr/small_hires896_v2/robustness

# B · build the D1 joint clear+degraded training set (offline, no RF-DETR internals)
step "B_make_d1_dataset" B_make_d1_dataset.log \
  $PY scripts/make_d1_dataset.py --src "$HIRES" --out "$D1DS" --frac 0.5

# C · train small@896 + D1 (headline robust model) — same regime as the baseline
#     (epochs 15 + early-stopping → converges ~epoch 11 like the clean run, fair compare)
step "C_train_small_d1" C_train_small_d1.log \
  $PY scripts/rfdetr_train.py --variant small --dataset-dir "$D1DS" --resolution $RES \
      --epochs 15 --early-stopping --num-workers 24 --batch-size 4 --grad-accum-steps 4 \
      --output-dir "$SD1"

# D · small+D1 clean-test eval (does D1 cost anything on clear? — honesty check)
step "D_eval_small_d1_clean" D_eval_small_d1_clean.log \
  $PY scripts/rfdetr_eval.py --weights "$SD1/checkpoint_best_total.pth" --variant small \
      --resolution $RES --split test

# E · small+D1 robustness table (reuse $DEG) — THE headline: D1 vs baseline on degraded
step "E_small_d1_robustness" E_small_d1_robustness.log \
  $PY scripts/rfdetr_robustness.py --weights "$SD1/checkpoint_best_total.pth" --variant small \
      --resolution $RES --degraded-root "$DEG" --plot --time-n 100 \
      --out-dir "$SD1/robustness"

# F · nano@896 baseline robustness (model exists; reuse $DEG) — variant-robustness row
step "F_nano_baseline_robustness" F_nano_baseline_robustness.log \
  $PY scripts/rfdetr_robustness.py --weights "$NANO" --variant nano --resolution $RES \
      --degraded-root "$DEG" --plot --time-n 100 \
      --out-dir runs/rfdetr/nano_hires896/robustness

# G · train nano@896 + D1 (the DEPLOYABLE robust model — Jetson story). Reuses $D1DS.
step "G_train_nano_d1" G_train_nano_d1.log \
  $PY scripts/rfdetr_train.py --variant nano --dataset-dir "$D1DS" --resolution $RES \
      --epochs 15 --early-stopping --num-workers 24 --batch-size 8 --grad-accum-steps 2 \
      --output-dir "$ND1"

# H · nano+D1 clean eval + robustness (deploy-model D1 gain)
step "H_eval_nano_d1_clean" H_eval_nano_d1_clean.log \
  $PY scripts/rfdetr_eval.py --weights "$ND1/checkpoint_best_total.pth" --variant nano \
      --resolution $RES --split test
step "H2_nano_d1_robustness" H2_nano_d1_robustness.log \
  $PY scripts/rfdetr_robustness.py --weights "$ND1/checkpoint_best_total.pth" --variant nano \
      --resolution $RES --degraded-root "$DEG" --plot --time-n 100 \
      --out-dir "$ND1/robustness"

# I · FILLER (only if budget remains): medium@896 variant-table completion
step "I1_train_medium" I1_train_medium.log \
  $PY scripts/rfdetr_train.py --variant medium --dataset-dir "$HIRES" --resolution $RES \
      --epochs 15 --early-stopping --num-workers 24 --batch-size 2 --grad-accum-steps 8 \
      --output-dir "$MEDOUT"
step "I2_eval_medium" I2_eval_medium.log \
  $PY scripts/rfdetr_eval.py --weights "$MEDOUT/checkpoint_best_total.pth" --variant medium \
      --resolution $RES --split test
step "I3_medium_robustness" I3_medium_robustness.log \
  $PY scripts/rfdetr_robustness.py --weights "$MEDOUT/checkpoint_best_total.pth" --variant medium \
      --resolution $RES --degraded-root "$DEG" --plot --time-n 100 \
      --out-dir "$MEDOUT/robustness"

say "ALL QUEUE STEPS ATTEMPTED. Robustness tables:"
find runs -path '*/robustness/robustness_table.csv' 2>/dev/null | sed 's/^/    /' | tee -a "$MASTER"
say "Queue finished. Master log: $MASTER"
