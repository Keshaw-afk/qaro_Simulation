# swarm_nav_rl: full documentation

## 1. Goal
Four differential-drive robots (TortoiseBot-style) each drive to their own endpoint in a rectangular forest-green arena with random
pink obstacles, avoiding obstacles and each other. Navigation is learned with RL. **One shared model** (spec "Alternative 1"):
each robot's sensor feed goes through the same network, which outputs that robot's movement command. Spec: `my_gifts/spec.txt`.

## 2. Results
Policy `weights/final_policy.npz` (PPO checkpoint at 7M steps, the best of run10). Evaluation layouts are seeds 0-199, never used in training (training uses seeds >= 1000).

| Evaluation | Success | Collision | Timeout | Mean time to goal | All 4 succeed |
|---|---|---|---|---|---|
| Go-to-goal baseline (no avoidance), 2D, 100 layouts | 43.2% | 56.8% | 0% | 8.5 s | 7% |
| Baseline + safety layer, 2D | 55.7% | 39.8% | 4.5% | 8.8 s | 14% |
| Final policy, 2D, 100 layouts | 91.5% | 4.0% | 4.5% | 11.3 s | 72% |
| **Final policy, Gazebo, 20 layouts x 4 robots** | **91.2%** (73/80) | **6.2%** (5/80) | **2.5%** (2/80) | **10.6 s** | **65%** (13/20) |

Gazebo run is closed loop with the real pipeline (camera tags, lidar, odometry); success/collision are scored from simulator ground truth.
Success = within 0.15 m of the goal; collision = chassis within 1 cm of an obstacle, wall or another robot; timeout = 700 policy steps (70 s).
Per-layout results are in `docs/gazebo_eval_20_layouts.log`. (`runs/gazebo_results.json` is overwritten by every evaluation run and now holds only a later single-layout rerun.) Training curve: `docs/run10_eval.csv`.
The 2D and Gazebo numbers agree to within a few points. The Gazebo sample is small (80 robot runs), so the 6.2% collision rate is uncertain by several points.
Causes of the 5 Gazebo collisions were not analysed; `rl/eval_gazebo.py` can print them (see Known limitations).

## 3. System overview
```
overhead camera -> AprilTag detection (tools/vision.py) -> robot poses, goal positions, velocities (frame differencing)
lidar (360 beams, 0.15-2 m) + wheel odometry (per robot)
        -> observation (59 numbers, rl/deploy.py) -> shared MLP policy (rl/deploy.py) -> one of 45 (speed, turn-rate) commands
        -> safety layer (rl/shield.py) may swap the command -> /rN/cmd_vel
```
Two environments share the same observation, action table and layout generator:
- **Gazebo Harmonic + ROS 2 Jazzy** (ground truth, used for the final evaluation; slow).
- **A numpy 2D vectorised environment** `rl/arena_env.py` (fast; used for training with 64 worlds = 256 agents).

## 4. Simulation (Gazebo)
- `config/robot.yaml`: robot parameters, each marked SOURCED (RigBetel Labs TortoiseBot files), DERIVED, or ESTIMATED (from the photos in `my_gifts/`). Notably wheel diameter 65 mm, wheel separation 155.6 mm, max speed 0.20 m/s, chassis radius 0.095 m (estimated), lidar 0.15-2.0 m at 10 Hz, centre of mass x = -0.03 (needed so the single-caster robot does not tip).
- `tools/robot_sdf.py`: robot SDF (coloured chassis, black wheels and lidar, AprilTag board on top, diff-drive plugin, gpu_lidar, odometry). Topics `/rN/cmd_vel`, `/rN/odom`, `/rN/scan`.
- `tools/gen_world.py`: world SDF: 4 x 3 m arena, 0.4 m grey walls, pink obstacles (height 0.4), goal tag boards (tags 4-7, 0.20 m, on the floor), overhead camera (2.5 m high, 1280x960). Robot tags are ids 0-3 (tag36h11).
- `tools/layout.py`: random layouts: 3-8 obstacles with sides 0.15-0.60 m. **No-softlock guarantee:** obstacles/walls are inflated by robot radius + margin; all starts and goals must lie in one connected free region. Goals must also be visible to the camera (not hidden behind a tall obstacle).
- `tools/gzsim.py`: lockstep stepping through gz-transport (`/world/arena/control`, multi_step with pause), sensor access. In evaluation runs the simulator stays paused between steps, so never press play there. Live mode (`SNR_LIVE=1`) is different: the world plays in real time (`GzSim.play()`), and the control loop only waits for the simulation clock.
- `tools/vision.py`: AprilTag detection (OpenCV aruco, multi-scale because some ids are missed at some scales) and pinhole conversion to world coordinates. Accuracy: 40/40 tags found, about 0.3 cm position error.
- Checks: `tools/check_robot.py`, `check_world.py`, `check_vision.py`, `tests/test_layout.py`, `tests/test_obs_parity.py`.

## 5. Policy
**Observation (59 numbers, per robot):**
| Part | Size | Content |
|---|---|---|
| Lidar | 36 | 36 sectors of 10 beams; min over 5 beams of each sector; range/2.0; readings below 0.15 m count as 0 (blind zone), above 2 m as 1 |
| Goal | 3 | min(distance, 4)/4, cos and sin of bearing to goal (from camera tag) |
| Own motion | 2 | forward speed/0.20, turn rate/2.5 (odometry) |
| Other robots | 12 | for each of the 3 others (ascending id): relative position/3 and relative velocity/0.4, clipped (from camera) |
| Previous action | 2 | last command, normalised |
| Own id | 4 | one-hot (so the shared policy knows which robot it is) |

`tests/test_obs_parity.py` checks that the training env and the deployment code (`rl/deploy.py`) build identical observations.

**Actions:** 45 discrete commands = 5 speeds {0, 0.05, 0.10, 0.15, 0.20} m/s x 9 turn rates {-2.0, -1.2, -0.7, -0.3, 0, 0.3, 0.7, 1.2, 2.0} rad/s (`rl/actions.py`). Discrete actions were used because a continuous Gaussian policy averages "go left" and "go right" into "go straight" in head-on situations.

**Network:** MLP 59 -> 256 -> 256 -> 45 (tanh). Deployment needs only numpy (`weights/final_policy.npz` holds the observation mean/variance and 3 linear layers).

**Why the lidar alone is not enough for robot-robot avoidance:** the scan plane (0.17 m) is above the chassis top (0.15 m), so a lidar sees only the other robot's 7 cm lidar head. Other robots' positions and velocities therefore come from the overhead camera.

## 6. Training
- `python rl/train.py RUN STEPS` (PPO, Stable-Baselines3): shared policy, 64 worlds x 4 robots, n_steps 128, batch 2048, 10 epochs, lr 3e-4 -> 1e-4, ent_coef 0.01, VecNormalize, an evaluation on 100 held-out layouts every 1M steps (writes `runs/RUN/eval.csv`, `best.npz`, `latest.npz`, `final.npz`).
- **Domain randomisation in the 2D env:** speed scale 0.9-1.1, actuator lag, 30% of robots with one step of command delay.
- **Reward:** 10 x (progress toward goal) - 0.01 per step - clearance penalties (obstacles/robots) - time-to-collision penalty - keep-right shaping - small turn-rate-change penalty; +10 on reaching the goal; -15 on collision. Robots stop on goal or collision.
- Run10 took 8M steps (about 2 h on this machine).

**Safety layer (`rl/shield.py`).** Rule-based, not learned. For the chosen command, it predicts the next 2 s of the robot's own motion and the other robots at the camera-measured constant velocity. If the chassis gap would fall below 6 cm it swaps in the nearest safe command (or the largest-gap command if none is safe). The policy was trained with the layer active, and it is applied in Gazebo too. It only vetoes robot-robot collision courses; goal seeking and obstacle avoidance are the policy's.
Effect in 2D: robot-robot collisions 18% -> 0%.

## 7. History of what was tried (so it is not repeated)
1. Runs 1-8, continuous-action PPO with various rewards/inputs: plateau at about 70% success, 25-30% collisions, mostly robot-robot. Two bugs found on the way: robot-robot clearance subtracted one radius instead of two; and lidar training/deployment parity.
2. DWA (classical planner) expert, `rl/expert.py`: about 78% success, 0% collisions: proves the task is fair. DAgger student (`rl/dagger.py`): 55% success, not competitive.
3. Run9, discrete actions: 71% success, 25% collisions.
4. Safety layer on run9 weights: 83% success, 8% collisions. Run10 (discrete + layer + retraining): 91.5% success, 4% collisions. Training curve flattened after about 4M steps.

## 8. How to run
### 8.0 Desktop app (easiest)
```
python3 swarm_gui.py          # or ./run_gui.sh
```
Settings: **Model** (every `.npz` in `weights/` and `runs/`, or Browse...), **Simulator** (Gazebo, or the fast 2D environment), **Arena**
(Standard 4 x 3 m, or Big 8 x 6 m; the 2D environment only knows the standard arena), **First layout / Number of layouts**,
and options: open the Gazebo window (plays in real time), safety layer on/off, save a video of each layout. An estimated duration is shown.
Layouts restart the simulator one after another; that is normal. Results and videos go to `runs/gui/`.

Safeguards against disturbing anything by accident:
- Each run uses its own Gazebo partition, so it cannot interfere with another evaluation or a simulator started elsewhere.
- **Stop this run** ends only this app's run (after a confirmation) and removes that run's simulator.
- There is no "kill everything" button. **Clean up leftovers...** lists only simulators that no running evaluation owns, and asks first.
- Settings are locked during a run; closing the window during a run asks; starting while another evaluation runs, or with the safety layer off, asks.
- Standard-arena layouts above 199 were used for training, so the app warns before using them.
In live mode the world plays in real time, so the play button shows playing, motion is continuous, and the Gazebo tools (e.g. lidar visualisation) work. Without the window, runs are stepped while paused for speed.

### 8.1 Command line
Big arena: prefix with `SNR_ARENA=big` (e.g. `SNR_ARENA=big SNR_FIRST=51 python3 rl/eval_gazebo.py weights/final_policy.npz 101 0`). Variables are listed at the top of `rl/eval_gazebo.py`.
Use the Ubuntu 24.04 distrobox container. Python env: `.venv` (created with `--system-site-packages`).
**Run these one at a time, not as one pasted block:** they run in order, and the training command takes about 2 hours, so everything after it waits.

Setup, once per terminal (Gazebo scripts need it):
```
source /opt/ros/jazzy/setup.bash
```
Tests (fast):
```
python3 tests/test_layout.py
.venv/bin/python tests/test_obs_parity.py
```
Gazebo evaluation of the final policy on 20 layouts (about 20 min; videos for the first 3 go to `runs/gazebo_seed*.mp4`):
```
python3 rl/eval_gazebo.py weights/final_policy.npz 20 3
```
Watch one layout in the Gazebo GUI, playing in real time (about 1 min):
```
SNR_LIVE=1 SNR_FIRST=13 python3 rl/eval_gazebo.py weights/final_policy.npz 14 0
```
One command does all of the following for a layout (pick one with `python3 tools/demo_layouts.py`, or `--hard --all` for the most obstacle-heavy ones): `bash tools/make_demo.sh 45 se live` (video to `runs/demo/`, then the live window).
Record a demo video from an angled camera (south-east `se` or south-west `sw` of the arena; about 1 min; writes `runs/demo_seed45.avi`).
Layout 45 shows every start and goal from both sides. OpenCV here cannot write H.264, so convert with ffmpeg for sharing:
```
SNR_DEMO=se SNR_FIRST=45 python3 rl/eval_gazebo.py weights/final_policy.npz 46 0
ffmpeg -i runs/demo_seed45.avi -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart runs/demo_seed45.mp4
```
Add `SNR_LIVE=1` to watch the same layout in the Gazebo window, which then opens on the same angled view (it also records the video).
Stop stray simulators:
```
bash tools/killsim.sh
```
Optional, retrain from scratch (about 2 h; builds `rl/pool_*.npz` on first run):
```
.venv/bin/python rl/train.py run11 8e6
```

Notes: `SNR_FIRST=k` with `N = k+1` runs only layout k. Run only one simulator at a time (several share topics and corrupt results). `runs/`, `rl/pool_*.npz` are not tracked.

## 9. Known limitations
- Gazebo evaluation is 80 robot runs, so rates carry several points of uncertainty.
- 6.2% of Gazebo runs ended in a collision and the cause was not diagnosed: `rl/eval_gazebo.py` has a diagnostic (added after the evaluation) that prints the object hit; it has not been run yet.
- The safety layer is a hand-written rule, not learned. Without it the policy collides far more.
- Robot size, chassis height, and lidar head size are estimates from photos, not measurements.
- Policy inputs rely on the overhead camera for goals and other robots; if a tag is not detected in a frame, deployment would need a fallback (not implemented).
- Real-robot transfer was not attempted.

## 10. Layout of the repository
```
config/robot.yaml     robot parameters      tools/    Gazebo world/robot generators, sim wrapper, vision, helper checks
rl/                   env, training, deploy, expert, DAgger, safety layer, Gazebo evaluator
tests/                layout and observation-parity tests      weights/  final policy      docs/  this file, training curve
worlds/               tag images and an early single-robot world
AGENTS.md CLAUDE.md CONTRIBUTING.md CHANGELOG.md Makefile requirements.txt   repository-level guides and helpers; swarm_gui.py = desktop app
```
