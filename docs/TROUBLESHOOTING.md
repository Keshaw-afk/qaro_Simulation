# Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: gz.transport13` | ROS not sourced. `source /opt/ros/jazzy/setup.bash`, then `python3 tools/check_env.py`. |
| `ModuleNotFoundError: gymnasium / torch / stable_baselines3` | You used the system python for 2D evaluation or training. Use `.venv/bin/python` (docs/SETUP.md section 4). |
| Results look random or a run hangs / crashes with "world control never advanced the clock" | Two simulators in the same `GZ_PARTITION` share topics. List them: `pgrep -af "[g]z sim"`. Stop stale ones by PID, or run concurrent jobs with different `GZ_PARTITION` values. |
| Gazebo window shows paused / jittery motion | Headless evaluations are stepped while paused by design. For watching, use live mode (`SNR_LIVE=1` or the app's "Open Gazebo window"), which plays in real time. Never run a headless evaluation in the same partition as a window you are watching. |
| "only N goal tags seen" assertion | A goal tag was not decoded or is hidden. Layouts keep goals camera-visible; if you changed camera height/arena size, keep pixels per metre constant (the big arena doubles the image size) and re-run `tools/check_vision.py`. |
| Robot drives backwards / tips over after editing the robot | Wheel link roll must be -1.5707963; centre of mass x must stay between axle and caster (`com_x: -0.03`). `tools/check_robot.py` checks both. |
| `pkill` killed my own shell (exit 144) | The command text contained the pattern. Use `tools/killsim.sh` or kill by PID. |
| 2D training/evaluation fails after changing layouts | Delete `rl/pool_*.npz` so they are rebuilt. |
| Headless rendering fails (no EGL/GL) | Gazebo needs off-screen rendering (GPU or Mesa). Check `gz sim -s --headless-rendering` on any world; install Mesa/EGL packages. |
| Slow runs | Gazebo with cameras is about 1 min/layout (2-3 min in the big arena); the 2D environment takes seconds. Do broad checks in 2D, confirm in Gazebo. |
