#!/usr/bin/env bash
# One video per controller (PPO, DQN seed 2, rule controller), each driving the same arenas in Gazebo, edited for viewing
# (title card per arena, idle stretches shortened, long runs fast-forwarded with an on-screen label; the overlay clock
# keeps true time). This is how instructor_artifacts/2026-11-11_controller_videos/ was made.
#   bash tools/make_comparison_videos.sh                        # designed arenas 9001-9005 (about 10 min)
#   bash tools/make_comparison_videos.sh "9001 9003" OUT_DIR     # other layouts / output folder
# Output: OUT_DIR/{ppo,dqn,rule}_arenas.mp4 (default OUT_DIR runs/demo/compare/final); raw clips in runs/demo/compare/NAME/.
# The three controllers record in parallel, each in its own GZ_PARTITION. Needs ROS 2 Jazzy + Gazebo (on a host without
# /opt/ros it re-runs itself in the distrobox SNR_CONTAINER, default "ubuntu"), and ffmpeg (host or container).
set -euo pipefail
cd "$(dirname "$0")/.."
ARENAS=${1:-"9001 9002 9003 9004 9005"}; OUT=${2:-runs/demo/compare/final}
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
    if ! command -v distrobox >/dev/null; then
        echo "error: needs ROS 2 Jazzy + Gazebo Harmonic (/opt/ros/jazzy/setup.bash not found) and no distrobox to switch into." >&2
        echo "See docs/SETUP.md section 0 (Ubuntu 24.04 native, distrobox, WSL2 or a VM)." >&2
        exit 1
    fi
    exec distrobox enter "${SNR_CONTAINER:-ubuntu}" -- bash -c "cd '$PWD' && bash tools/make_comparison_videos.sh '$ARENAS' '$OUT'"
fi
declare -A SPEC=([ppo]="" [dqn]="npz:runs/benchmark/dqn/final_s2/best.npz" [rule]="rule")
declare -A LABEL=([ppo]="PPO" [dqn]="DQN" [rule]="rule controller")
title() { python3 -c "import sys; sys.path.insert(0, 'tools'); import designed_layouts as D; k=int(sys.argv[1]); print(f'designed arena {k} ({D.DESIGNS[k][0]})' if k in D.DESIGNS else f'layout {k}')" "$1"; }
for m in ppo dqn rule; do
    (
        for k in $ARENAS; do
            GZ_PARTITION=snr_compare_$m SNR_SPEC="${SPEC[$m]}" SNR_NAME=compare/$m SNR_CAPTION="${LABEL[$m]}, $(title $k)" \
                bash tools/make_demo.sh "$k" se 2>&1 | grep -E "^seed" | sed "s/^/$m /"
        done
    ) &
done
wait
mkdir -p "$OUT"
python3 tools/edit_comparison_video.py runs/demo/compare "$OUT" $ARENAS
FFMPEG=(ffmpeg); command -v ffmpeg >/dev/null || FFMPEG=(distrobox-host-exec ffmpeg)
for m in ppo dqn rule; do
    "${FFMPEG[@]}" -y -loglevel error -i "$OUT/$m.avi" -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "$OUT/${m}_arenas.mp4" \
        && rm "$OUT/$m.avi" && echo "video: $PWD/$OUT/${m}_arenas.mp4"
done
