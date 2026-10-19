# swarm_nav_rl: 4 robots, one shared RL policy, Gazebo Harmonic + ROS 2 Jazzy

Four differential-drive TortoiseBot-style robots drive to per-robot goals in a random-obstacle arena. One shared policy: each robot's
sensors (lidar, odometry, overhead-camera AprilTag poses/goals) go through the same network to that robot's drive command.

## Result (best policy `runs/final_policy.npz`, PPO 7M steps)
| Test | Success | Collision | Timeout | Time to goal | All 4 succeed |
|---|---|---|---|---|---|
| 2D training env, 100 held-out layouts | 91.5% | 4.0% | 4.5% | 11.3 s | 72% |
| **Gazebo, 20 held-out layouts x 4 robots, camera-in-the-loop** | **91.2%** | **6.2%** | **2.5%** | **10.6 s** | **65%** |

Go-to-goal baseline (no avoidance): 43% success. Videos: `runs/gazebo_seed*.mp4`; live screenshots: `runs/screenshots/`.

## Quick start
```
python3 swarm_gui.py        # desktop app: pick a model, choose layouts, run in Gazebo (live window) or the fast 2D env
```
Needs ROS 2 Jazzy + Gazebo Harmonic: see `docs/SETUP.md` (setup recipe, `tools/check_env.py` to verify a machine). Full docs: `docs/DOCUMENTATION.md` (what and why), `docs/CODE_GUIDE.md` (how the code is organised).

## How it works
- `tools/` world/robot SDF generators, layout generator (no-softlock), Gazebo lockstep wrapper (`gzsim.py`), AprilTag vision (`vision.py`).
- `rl/arena_env.py` fast numpy 2D env mirroring Gazebo; `rl/train.py` PPO (SB3), 45 discrete (speed, turn-rate) actions.
- `rl/shield.py` rule-based inter-robot safety layer (predicts 2 s ahead using camera velocities, vetoes robot-robot collision courses).
  The RL policy still does goal-seeking and obstacle avoidance. It was trained with the layer active and is deployed with it.
- `rl/deploy.py` numpy inference; `rl/eval_gazebo.py WEIGHTS.npz N [VIDEOS]` closed-loop Gazebo evaluation
  (`SNR_LIVE=1 SNR_FIRST=k` opens the Gazebo GUI on layout k; do not press play, the sim is stepped in lockstep).

## Findings
- Plain PPO plateaued at ~70% success / 25-30% collisions, mostly robot-robot; a DWA expert gets 0% collisions, so the task was fair.
- DAgger imitation (55%) was not competitive. Discrete actions helped a little (71%); the safety layer + retraining gave 91%.
- Lidar only sees other robots' small lidar heads, so robot-robot avoidance needs camera positions/velocities.
