#!/usr/bin/env bash
# One robot live in the Gazebo window + the policy dashboard (what the model sees and decides, refreshed every PERIOD s).
#   bash experiments/live_dashboard.sh SEED [ROBOT=0] [VW=camera|cmdmodel|odom] [PERIOD=1.0]
# On a distrobox host without ROS it re-runs itself inside the container (SNR_CONTAINER, default "ubuntu").
# The dashboard window stays open after the run (showing the final status) until you close it.
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${1:?usage: live_dashboard.sh SEED [ROBOT] [camera|cmdmodel|odom] [PERIOD]}; ROBOT=${2:-0}; VW=${3:-camera}; PERIOD=${4:-1.0}
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
    exec distrobox enter "${SNR_CONTAINER:-ubuntu}" -- bash -c "cd '$PWD' && bash experiments/live_dashboard.sh $SEED $ROBOT $VW $PERIOD"
fi
set +u; source /opt/ros/jazzy/setup.bash; set -u
export GZ_PARTITION=${GZ_PARTITION:-snr_dash}
python3 experiments/dashboard.py "$PERIOD" &
DASH_PID=$!
sleep 2
SNR_DASH=1 SNR_LIVE=1 SNR_FIRST=$SEED SNR_ROBOT=$ROBOT SNR_VW=$VW SNR_OUT_DIR=runs/dash_live \
    python3 experiments/eval_gazebo_single.py weights/final_policy.npz $((SEED + 1))
echo "per-step data: $PWD/runs/dash_live/dash/seed${SEED}_robot${ROBOT}.jsonl; close the dashboard window to finish"
wait $DASH_PID || true
