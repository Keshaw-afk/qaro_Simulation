#!/usr/bin/env bash
# Demo of one layout from the angled camera: records runs/demo/demo_seed{SEED}.mp4 (exact, stepped), then with "live"
# plays the same layout in real time in the Gazebo window, opened on the same view. Pick layouts with tools/demo_layouts.py.
#   bash tools/make_demo.sh SEED [se|sw] [live|live-only|watch]
#     live-only: skip the stepped recording, play it in the window (still saves that live run's video)
#     watch:     only the Gazebo window (angled view, coloured goal discs), nothing recorded or saved
# Another controller: SNR_SPEC=SPEC (as in benchmark/evaluate_all.py, e.g. npz:runs/benchmark/dqn/final_s2/best.npz or rule)
# and SNR_NAME=NAME; videos then go to runs/demo/NAME/ (default: PPO, weights/final_policy.npz, runs/demo/).
# SNR_CAPTION sets the overlay text after "4 robots, " (default "one shared RL policy").
# On a distrobox host without ROS it re-runs itself inside the container (SNR_CONTAINER, default "ubuntu").
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${1:?usage: make_demo.sh SEED [se|sw] [live|live-only]}; SIDE=${2:-se}; LIVE=${3:-}
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
    exec distrobox enter "${SNR_CONTAINER:-ubuntu}" -- bash -c "cd '$PWD' && SNR_SPEC='${SNR_SPEC:-}' SNR_NAME='${SNR_NAME:-}' SNR_CAPTION='${SNR_CAPTION:-}' bash tools/make_demo.sh $SEED $SIDE $LIVE"
fi
set +u; source /opt/ros/jazzy/setup.bash; set -u
export GZ_PARTITION=${GZ_PARTITION:-snr_demo}
SPEC=${SNR_SPEC:-}; D=runs/demo${SNR_NAME:+/$SNR_NAME}
PY=python3; [ -x .venv/bin/python ] && PY="$PWD/.venv/bin/python"
run() {  # OUT_DIR [extra env]: one layout with PPO (rl/eval_gazebo.py) or with SPEC (benchmark/demo_bench.py)
    local out=$1; shift
    if [ -z "$SPEC" ]; then env "$@" SNR_DEMO=$SIDE SNR_FIRST=$SEED SNR_OUT_DIR=$out python3 rl/eval_gazebo.py weights/final_policy.npz $((SEED + 1)) 0
    else env "$@" SNR_DEMO=$SIDE SNR_OUT_DIR=$out $PY benchmark/demo_bench.py "$SPEC" $SEED; fi
}
if pgrep -f "[g]z sim" >/dev/null; then echo "note: other simulators are running (pgrep -af '[g]z sim'); this run uses GZ_PARTITION=$GZ_PARTITION"; fi
FFMPEG=(ffmpeg); command -v ffmpeg >/dev/null || FFMPEG=(distrobox-host-exec ffmpeg)  # the container may lack ffmpeg; the host has it
to_mp4() {  # DIR: demo_seed$SEED.avi -> H.264 .mp4 (plays everywhere); keeps the .avi if no encoder is available
    if "${FFMPEG[@]}" -y -loglevel error -i "$1/demo_seed$SEED.avi" -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart "$1/demo_seed$SEED.mp4"; then
        rm "$1/demo_seed$SEED.avi"; echo "video: $PWD/$1/demo_seed$SEED.mp4"
    else
        echo "no H.264 ffmpeg found: video left as $PWD/$1/demo_seed$SEED.avi"
    fi
}

if [ "$LIVE" = watch ]; then
    run /tmp SNR_LIVE=1 SNR_RECORD=0
    exit 0
fi

if [ "$LIVE" != live-only ]; then
    mkdir -p $D
    run $D
    to_mp4 $D
fi

if [ "$LIVE" = live ] || [ "$LIVE" = live-only ]; then  # real time, so the run can differ slightly from the recorded one; its own video goes to runs/demo/live/
    mkdir -p $D/live
    run $D/live SNR_LIVE=1
    to_mp4 $D/live
fi
