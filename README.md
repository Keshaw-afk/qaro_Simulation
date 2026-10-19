# qaro_Simulation: 4 robots, one shared RL policy (Gazebo Harmonic + ROS 2 Jazzy)

Four differential-drive TortoiseBot-style robots drive to their own goals in a random-obstacle arena, avoiding obstacles and each other.
**One shared policy**: each robot's own sensors (lidar, wheel odometry, plus overhead-camera AprilTag positions of goals and the other robots)
go through the same network, which outputs that robot's drive command. Trained with PPO in a fast 2D twin of the arena, evaluated in Gazebo.

## Results (policy `weights/final_policy.npz`)
| Test | Success | Collision | Timeout | Time to goal | All 4 succeed |
|---|---|---|---|---|---|
| Go-to-goal baseline (no avoidance), 2D, 100 layouts | 43.2% | 56.8% | 0% | 8.5 s | 7% |
| 2D training environment, 100 held-out layouts | 91.5% | 4.0% | 4.5% | 11.3 s | 72% |
| **Gazebo, 20 held-out layouts x 4 robots, camera in the loop** | **91.2%** (73/80) | **6.2%** (5/80) | **2.5%** (2/80) | **10.6 s** | **65%** |

More Gazebo runs (standard arena layouts 1-50 and a big 8 x 6 m arena, layouts 51-100) are recorded in `docs/results/`.
Details, caveats and the history of what was tried: `docs/DOCUMENTATION.md`.

## Quick start (Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic)
```
source /opt/ros/jazzy/setup.bash
python3 tools/check_env.py                     # verify the machine (see docs/SETUP.md to install what is missing)
make test                                      # fast tests, no simulator
python3 swarm_gui.py                           # desktop app: choose model, arena (standard / big), layouts, live window
make eval-small FIRST=1 N=3                    # or from the command line: Gazebo on layouts 1-3
make eval-big FIRST=51 N=3                     # the same in the big 8 x 6 m arena
```
Training and the 2D evaluation need a virtualenv (`docs/SETUP.md`, section 4): `make train RUN=run11 STEPS=8e6`, `make eval-2d`.

## How it works
- `tools/`: world and robot SDF generators, random layouts with a no-softlock guarantee, lockstep Gazebo interface, AprilTag vision.
- `rl/arena_env.py`: fast numpy 2D environment mirroring Gazebo. `rl/train.py`: PPO (Stable-Baselines3) with 45 discrete (speed, turn-rate) actions.
- `rl/shield.py`: rule-based inter-robot safety layer. It predicts 2 s ahead from camera-measured velocities and vetoes robot-robot collision courses.
  The learned policy does goal seeking and obstacle avoidance; it was trained with the layer active and is deployed with it.
- `rl/deploy.py`: numpy inference and the sensor-to-observation code. `rl/eval_gazebo.py`: closed-loop Gazebo evaluation.
- `swarm_gui.py`: desktop launcher with safeguards (isolated simulator per run, confirmations, no kill-all).

## For contributors and AI agents
Start with **`AGENTS.md`** (commands, rules that prevent known failure modes, definition of done), then `CONTRIBUTING.md`,
`docs/CODE_GUIDE.md` and `docs/TROUBLESHOOTING.md`. Index of all documents: `docs/README.md`. Changes: `CHANGELOG.md`.

## Status and limits
Finished and evaluated, not tested on other machines or on real robots. The safety layer is a hand-written rule, not learned. Robot dimensions not given
were estimated from photos. About 6% of Gazebo runs end in a collision whose cause has not been analysed.

## License and credits
MIT License, Copyright (c) 2026 Keshaw Tiwari: free to use, modify and share, as long as the copyright notice (`LICENSE`) is kept.
Robot dimensions are based on the public RigBetel Labs TortoiseBot description (`config/robot.yaml` marks each value as sourced, derived or estimated);
no code from that project is included. Built on Gazebo, ROS 2, OpenCV, Stable-Baselines3 and PyTorch, which keep their own licenses.
