# Setup on a new machine

Developed and tested on Ubuntu 24.04 (inside a distrobox container) with ROS 2 Jazzy and Gazebo Harmonic (gz sim 8.15) installed from the
ROS Jazzy packages. It has **not** been tried on another machine, so treat this as the expected recipe and report anything that differs.

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
`--system-site-packages` lets the venv reuse the apt-installed scipy/numpy/OpenCV. On first use the layout pools `rl/pool_*.npz` are generated
(a few minutes). Python 3.12 is what was used.

## Known portability limits
- `/opt/ros/jazzy/setup.bash` is assumed (the app and docs); other ROS locations need edits to `ROS_SETUP` in `swarm_gui.py`.
- `tools/screenshot.py` is GNOME/Wayland specific and optional. The app uses `xdg-open` to open the results folder.
- Other Gazebo/ROS versions (e.g. Gazebo Ionic) use different Python module names (`gz.transport13`, `gz.msgs10`) and would need changes in `tools/gzsim.py`.
