#!/usr/bin/env bash
# Demo of one layout from the angled camera: records runs/demo/demo_seed{SEED}.mp4 (exact, stepped), then with "live"
# plays the same layout in real time in the Gazebo window, opened on the same view. Pick layouts with tools/demo_layouts.py.
#   bash tools/make_demo.sh SEED [se|sw] [live]
# On a distrobox host without ROS it re-runs itself inside the container (SNR_CONTAINER, default "ubuntu").
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${1:?usage: make_demo.sh SEED [se|sw] [live]}; SIDE=${2:-se}; LIVE=${3:-}
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
    exec distrobox enter "${SNR_CONTAINER:-ubuntu}" -- bash -c "cd '$PWD' && bash tools/make_demo.sh $SEED $SIDE $LIVE"
fi
set +u; source /opt/ros/jazzy/setup.bash; set -u
export GZ_PARTITION=${GZ_PARTITION:-snr_demo}
if pgrep -f "[g]z sim" >/dev/null; then echo "note: other simulators are running (pgrep -af '[g]z sim'); this run uses GZ_PARTITION=$GZ_PARTITION"; fi
FFMPEG=(ffmpeg); command -v ffmpeg >/dev/null || FFMPEG=(distrobox-host-exec ffmpeg)  # the container may lack ffmpeg; the host has it
to_mp4() {  # DIR: demo_seed$SEED.avi -> H.264 .mp4 (plays everywhere); keeps the .avi if no encoder is available
    if "${FFMPEG[@]}" -y -loglevel error -i "$1/demo_seed$SEED.avi" -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart "$1/demo_seed$SEED.mp4"; then
        rm "$1/demo_seed$SEED.avi"; echo "video: $PWD/$1/demo_seed$SEED.mp4"
    else
        echo "no H.264 ffmpeg found: video left as $PWD/$1/demo_seed$SEED.avi"
    fi
}

mkdir -p runs/demo
SNR_DEMO=$SIDE SNR_FIRST=$SEED SNR_OUT_DIR=runs/demo python3 rl/eval_gazebo.py weights/final_policy.npz $((SEED + 1)) 0
to_mp4 runs/demo

if [ "$LIVE" = live ]; then  # real time, so the run can differ slightly from the recorded one; its own video goes to runs/demo/live/
    mkdir -p runs/demo/live
    SNR_LIVE=1 SNR_DEMO=$SIDE SNR_FIRST=$SEED SNR_OUT_DIR=runs/demo/live python3 rl/eval_gazebo.py weights/final_policy.npz $((SEED + 1)) 0
    to_mp4 runs/demo/live
fi
