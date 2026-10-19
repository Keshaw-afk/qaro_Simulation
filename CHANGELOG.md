# Changelog

Milestones, newest first. Tags: `v0.1-one-robot`, `v0.2-four-robots`, `v0.3-rl-policy`, `v0.4-gui`.

## Unreleased
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
