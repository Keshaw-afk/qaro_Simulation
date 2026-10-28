# Changelog

Milestones, newest first. Tags: `v0.1-one-robot`, `v0.2-four-robots`, `v0.3-rl-policy`, `v0.4-gui`, `v1.0-navigation-demo`.

## Unreleased

## v1.0-navigation-demo: Robots Navigation and Collision Avoidance Demonstration
First complete version: four robots reach their goals in Gazebo with the PPO policy and safety layer (standard arena, layouts 1-50: 89.8% success, 6.1% collision, 4.1% timeout; see `docs/results/SUMMARY.md`).
- Instructor demo kit (`instructor_artifacts/`): demo scripts, checked layouts, ROS/Gazebo info, training curve.
- Run log (`SNR_LOG=1`, `rl/run_log.py`): once per simulated second prints each robot's ground-truth position, heading, speed, distance to goal and collision signals (smallest clearance, near misses, safety-layer interventions); every step goes to `log_seed{k}.csv`. Read-only, the policy path is unchanged.
- `eval_gazebo.py` exits without interpreter teardown after writing results: a late gz-transport message during shutdown segfaulted it once (live demo, layout 8), after all work was done.
- `tools/make_demo.sh SEED [se|sw] [live]` (record + convert + optional live run, re-runs itself in the distrobox from the host) and `tools/demo_layouts.py` (ranks layouts for demos by camera visibility; `--hard` by obstacles in the robots' straight paths and path crossings). Live mode with `SNR_DEMO` opens the Gazebo window on the demo view.
- Demo videos from an angled 3/4 camera (`SNR_DEMO=se|sw`, `rl/demo_video.py`): real-time video by simulation timestamps, status overlay, coloured goal discs; all visual only, normal evaluation worlds are unchanged. Layout 45 is a good demo (8 obstacles, every start and goal visible, all four succeed).
- Fix for a clean apt Gazebo + ROS install with a VPN: pin `GZ_IP`, restore `GZ_CONFIG_PATH`, keep the simulator log and show its tail on a startup timeout (found by debugging a friend's laptop over SSH).
- Big arena profile (`SNR_ARENA=big`, 8 x 6 m); evaluation of the final policy on layouts 1-50 (standard) and 51-100 (big) in `docs/results/`.
- Isolated Gazebo partitions so several runs coexist; safer desktop app (confirmations, own-run Stop, no kill-all).
- Live mode plays Gazebo in real time (previously lockstep-paused stepping looked jittery in the GUI).
- Repository preparation: `AGENTS.md`, `CONTRIBUTING.md`, `Makefile`, CI for the fast tests, setup and troubleshooting guides, shield test.

## v0.4-gui
- Native Tk desktop app (`swarm_gui.py`), 2D evaluation CLI, safety-layer toggle, code guide.

## v0.3-rl-policy
- Final policy: discrete 45-action PPO trained with a rule-based inter-robot safety layer. Gazebo, 20 held-out layouts: 91.2% success, 6.2% collision, 2.5% timeout.
- DWA expert and DAgger experiments (not used by the final policy); env bug fix (robot-robot clearance used one radius instead of two).

## v0.2-four-robots
- Four-robot arena, no-softlock layout generator, lockstep Gazebo interface, overhead camera and AprilTag vision (0.3 cm mean error), 2D training environment.

## v0.1-one-robot
- TortoiseBot-style robot model, headless gpu_lidar, drive checks (odometry matches ground truth).
