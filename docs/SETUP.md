# Setup on a new machine

Developed and tested on Ubuntu 24.04 (inside a distrobox container) with ROS 2 Jazzy and Gazebo Harmonic (gz sim 8.15) installed from the
ROS Jazzy packages. It has **not** been tried on another machine, so treat this as the expected recipe and report anything that differs.

## 0. Pick your route (any operating system)
Two halves of the project have very different needs:
- **2D half** (benchmark tables, 2D evaluation, training, tests, designed-layout checks): plain Python 3.12 and
  `pip install -r requirements.txt`. No ROS, no Gazebo, no GPU.
- **Gazebo half** (Gazebo evaluation, live window, demo videos, desktop app): Ubuntu 24.04 with ROS 2 Jazzy and Gazebo
  Harmonic, and a display only for live windows.

| Your system | 2D half | Gazebo half | Status |
|---|---|---|---|
| Ubuntu 24.04 (native or VM) | venv, section 4 | sections 1-3 | expected to work; the recipe below |
| Other Linux (Fedora, Arch, Debian, ...) | venv on the host, or inside the container | Ubuntu 24.04 [distrobox](https://distrobox.it): `distrobox create -n ubuntu -i ubuntu:24.04`, `distrobox enter ubuntu`, then sections 1-4 inside it. `tools/make_demo.sh` and `tools/make_comparison_videos.sh` switch into the container `SNR_CONTAINER` (default `ubuntu`) by themselves. | **verified** (Fedora 44 host, the development setup) |
| Windows 10/11 | Python 3.12 + venv (commands below use `/`; in PowerShell use `.venv\Scripts\python`) | WSL2 with Ubuntu 24.04, then sections 1-4 inside WSL (WSLg shows the Gazebo window) | not tested |
| macOS (Intel or Apple silicon) | Python 3.12 + venv | no native ROS 2 Jazzy + Gazebo Harmonic binaries for this setup: use an Ubuntu 24.04 VM (UTM, Parallels; arm64 Ubuntu has Jazzy packages) | not tested |

2D half only, any OS (verified on a clean venv without ROS or system packages, Python 3.12):
```
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build is enough
.venv/bin/pip install -r requirements.txt
.venv/bin/python tests/test_layout.py && .venv/bin/python tests/test_shield.py && .venv/bin/python tests/test_obs_parity.py
.venv/bin/python benchmark/make_results.py      # recomputes every benchmark table (benchmark/RESULTS.md, Reproduce)
```
`rl/pool_train.npz` and `rl/pool_eval.npz` are in the repository: never delete them (see `AGENTS.md` rule 6).
A pip-only environment reproduces the published 2D outcomes for PPO and the rule controller exactly and DQN on 399 of 400
test robots (floating-point differences between BLAS builds); for bit-identical numbers use the exact environment listed in
`benchmark/RESULTS.md` (Reproduce).

## 1. System packages (Ubuntu 24.04)
```
sudo apt install ros-jazzy-ros-base ros-jazzy-ros-gz-sim ros-jazzy-gz-transport-vendor ros-jazzy-gz-sim-vendor \
                 python3-opencv python3-scipy python3-yaml python3-numpy python3-tk python3-venv
```
`ros-jazzy-gz-transport-vendor` provides the Python bindings (`gz.transport13`, `gz.msgs10`) and `ros-jazzy-gz-sim-vendor` the `gz` command.
A standalone Gazebo Harmonic install should also work if the same Python modules and the `gz` command are available (not tested).
Gazebo must be able to render off-screen (`--headless-rendering`, needs working EGL/OpenGL: a GPU or Mesa). Linux only (the app reads `/proc`).

## 2. Check the machine
```
source /opt/ros/jazzy/setup.bash
python3 tools/check_env.py
```
All "needed" lines must say OK.

## 3. Run the trained policy (no training dependencies needed)
```
source /opt/ros/jazzy/setup.bash
python3 swarm_gui.py                                   # desktop app
python3 rl/eval_gazebo.py weights/final_policy.npz 3 0 # or from the command line: layouts 0-2
```

## 4. Training and the 2D evaluation (optional)
```
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python rl/train.py run11 8e6
```
`--system-site-packages` lets the venv reuse the apt-installed scipy/numpy/OpenCV. The layout pools `rl/pool_*.npz` come with the
repository (the training pool cannot be regenerated identically: never delete it). Python 3.12 is what was used.

## Known portability limits
- `/opt/ros/jazzy/setup.bash` is assumed (the app and docs); other ROS locations need edits to `ROS_SETUP` in `swarm_gui.py`.
- Other Gazebo/ROS versions (e.g. Gazebo Ionic) use different Python module names (`gz.transport13`, `gz.msgs10`) and would need changes in `tools/gzsim.py`.
