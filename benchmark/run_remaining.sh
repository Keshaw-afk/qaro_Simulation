#!/usr/bin/env bash
# Unattended rest of the benchmark pipeline (survives the Claude session closing). Run inside the ubuntu distrobox:
#   setsid nohup bash benchmark/run_remaining.sh > runs/benchmark/logs/run_remaining.log 2>&1 &
# Steps (each skipped if its output already exists, so it can be re-run safely):
#   1. rule tuning finished (or re-run if its process died without output) -> 2D test evaluation of `rule`
#   2. exploratory fine tabq run finished -> 2D evaluation (reported as exploratory)
#   3. Gazebo, layouts 100-119: PPO
#   4. DQN weights copied from Colab/Drive into runs/benchmark/dqn/final_s{1,2,3}/best.npz -> 2D evaluation of each seed,
#      Gazebo of the validation-chosen seed (waits up to 6 h for the files)
#   5. Gazebo of the best non-privileged rival among rule / tabq (by test success, safety layer on)
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset GITHUB_PERSONAL_ACCESS_TOKEN
PY="$PWD/.venv/bin/python"
R=benchmark/results
log() { echo "[$(date +%H:%M:%S)] $*"; }

# 1. rule
while [ ! -f $R/rule_best_params.json ]; do
  if ! pgrep -f "[r]ule.py tune" > /dev/null; then
    log "rule tuning not running and no result: running it (6 processes)"
    (cd benchmark && $PY rule.py tune 6 > ../runs/benchmark/logs/rule_tune_rerun.log 2>&1)
  else
    sleep 120
  fi
done
log "rule params: $(cat $R/rule_best_params.json | head -c 300)"
[ -f $R/raw/rule__test__shield1.json ] || (cd benchmark && $PY evaluate_all.py rule=rule > ../runs/benchmark/logs/eval_rule.log 2>&1)
log "rule evaluated"; grep "test sh" runs/benchmark/logs/eval_rule.log | cut -c1-200

# 2. exploratory fine tabq
F=runs/benchmark/tabq/explore_fine_fg_s1
while pgrep -f "[e]xplore_fine_fg_s1" > /dev/null; do sleep 120; done
if [ -f $F/best.npz ] && [ ! -f $R/raw/tabq_fine_explore__test__shield1.json ]; then
  (cd benchmark && $PY evaluate_all.py tabq_fine_explore=tabq:../$F/best.npz > ../runs/benchmark/logs/eval_tabq_fine_explore.log 2>&1)
  log "fine tabq evaluated"; grep "test sh" runs/benchmark/logs/eval_tabq_fine_explore.log | cut -c1-200
fi

# Gazebo helper: NAME SPEC (layouts 100-119, own partition, ROS sourced)
gz() {
  [ -f $R/gazebo__$1.json ] && grep -q '"summary"' $R/gazebo__$1.json && { log "gazebo $1 already done"; return; }
  log "gazebo $1 ($2) start"
  (set +u; source /opt/ros/jazzy/setup.bash && GZ_PARTITION=snr_bench_$1 $PY benchmark/eval_gazebo_bench.py "$1" "$2" 100 120 \
     > runs/benchmark/logs/gazebo_$1.log 2>&1)
  log "gazebo $1 done: $(tail -1 runs/benchmark/logs/gazebo_$1.log | cut -c1-250)"
}

# 3. PPO in Gazebo
gz ppo npz:weights/final_policy.npz

# 4. DQN
D=runs/benchmark/dqn
for i in $(seq 1 180); do
  [ -f $D/final_s1/best.npz ] && [ -f $D/final_s2/best.npz ] && [ -f $D/final_s3/best.npz ] && [ -f $D/final_s3/DONE ] && break
  sleep 120
done
if [ -f $D/final_s3/DONE ]; then
  for s in 1 2 3; do
    [ -f $R/raw/dqn_s${s}__test__shield1.json ] || (cd benchmark && $PY evaluate_all.py dqn_s$s=npz:../$D/final_s$s/best.npz > ../runs/benchmark/logs/eval_dqn_s$s.log 2>&1)
    grep "test sh" runs/benchmark/logs/eval_dqn_s$s.log | cut -c1-200
  done
  # validation-chosen seed: highest validation score of the kept checkpoint (never test)
  CH=$($PY -c "
import csv
best=max((max(float(r['val_score']) for r in csv.DictReader(open('$D/final_s%d/log.csv'%s))), s) for s in (1,2,3))
print(best[1])")
  log "DQN validation-chosen seed: $CH"
  gz dqn_s$CH npz:$D/final_s$CH/best.npz
else
  log "DQN weights not found after waiting; skipped (copy them and re-run this script)"
fi

# 5. best of rule / tabq (validation-chosen tabq seed s3) by test success with the safety layer
SECOND=$($PY -c "
import json
s={n: json.load(open('$R/raw/%s__test__shield1.json'%n)) for n in ('rule','tabq_s3')}
import numpy as np
print(max(s, key=lambda n: (np.asarray(s[n]['outcome'])==1).mean()))")
case $SECOND in
  rule) gz rule rule ;;
  tabq_s3) gz tabq_s3 tabq:runs/benchmark/tabq/final_s3/best.npz ;;
esac
log "ALL DONE"
