# Code guide (for working on the project later)

Read `DOCUMENTATION.md` first for what the system does. This file is about the code: where things live, how data flows, and what to
check when you change something.

## Module map
| File | Role | Needs |
|---|---|---|
| `swarm_gui.py` | Desktop launcher (Tk). Starts the evaluation scripts as subprocesses and shows their output. Contains no RL code. | python3 + tkinter |
| `rl/actions.py` | The 45-entry action table (speed, turn rate). **Single source of truth**: env, deploy, shield all import it. | numpy |
| `rl/arena_env.py` | Fast 2D vectorised environment (training). Observation, reward, dynamics, domain randomisation, evaluation bookkeeping (`outcome`, `cause`). | numpy, gymnasium, SB3 |
| `rl/shield.py` | Safety layer: `apply(pos, yaw, other_pos, other_vel, action_idx) -> action_idx`. Pure numpy, used by the 2D env and by Gazebo evaluation. | numpy |
| `rl/deploy.py` | Everything needed to run a trained policy outside training: `Policy` (numpy MLP), `obs_from_sensors` (builds the 59-number observation from real sensor data), `lidar_features`. | numpy |
| `rl/train.py` | PPO training script (`python rl/train.py RUN STEPS`). Writes `runs/RUN/{eval.csv,best.npz,latest.npz,final.npz}`. | .venv |
| `rl/evaluate.py` | `run_eval` (2D protocol evaluation) and the go-to-goal baseline. | .venv |
| `rl/eval2d.py` | Command-line wrapper around `run_eval`. | .venv |
| `rl/eval_gazebo.py` | Closed-loop Gazebo evaluation: starts the simulator per layout, runs the full camera/lidar/odometry pipeline, scores with ground truth. Options by environment variables (below). | ROS sourced |
| `rl/pool.py` | Pre-generates validated layouts into `rl/pool_{train,eval}.npz` (seeds >= 1000 train, 0-199 eval). Rebuild (delete the files) if layout generation changes. | scipy |
| `rl/expert.py`, `rl/dagger.py` | DWA expert and DAgger imitation (experiments; not used by the final policy). | .venv |
| `tools/layout.py` | Random layout generator with the no-softlock guarantee and camera-visibility rule. Shared by Gazebo and the 2D env. | scipy |
| `tools/robot_sdf.py`, `tools/gen_world.py` | Generate the robot and the world SDF files. | |
| `tools/gzsim.py` | Lockstep Gazebo control and sensor reads over gz-transport. | ROS sourced |
| `tools/vision.py` | AprilTag detection and image -> world coordinates. | OpenCV |
| `tools/killsim.sh` | Stop all `gz sim` processes safely (see pitfalls). | |
| `tests/` | `test_layout.py` (generator guarantees), `test_obs_parity.py` (training and deployment observations/actions identical). | |

## Data flow in a Gazebo evaluation (`rl/eval_gazebo.py: run_layout`)
1. `layout.generate_layout(seed)` -> `gen_world.world_sdf` -> write SDF -> start `gz sim -s --headless-rendering`.
2. Each policy step (0.1 s of simulated time = 20 physics steps):
   camera image -> `vision.detect` -> robot poses and goal positions; velocities by differencing poses ->
   per robot: `deploy.obs_from_sensors(lidar, own pose, goal, others, velocities, odometry, previous action, own id)` ->
   `Policy.index` (argmax of 45 logits) -> `shield.apply` -> command from `actions` table -> `/rN/cmd_vel`.
3. Ground-truth poses from the simulator decide success / collision; robots that finish are stopped.

Environment variables of `eval_gazebo.py`: `SNR_LIVE=1` open the Gazebo GUI and run at about real time; `SNR_FIRST=k` start at layout k
(layouts k .. N-1 run); `SNR_SHIELD=0` disable the safety layer. Arguments: `WEIGHTS.npz N_END [N_VIDEOS]`.

## Common changes and what else to touch
- **Change the observation** (`arena_env.py` obs builder *and* `deploy.obs_from_sensors` *and* `OBS_DIM`): run `tests/test_obs_parity.py`; old weights become invalid; retrain.
- **Change the actions**: edit only `rl/actions.py`; the network output size changes, so retrain; run the parity test.
- **Change the reward**: `ArenaVecEnv.step_wait` in `arena_env.py`. Keep terms small relative to the +10 goal / -15 collision, retrain and compare using `docs/run10_eval.csv` as the reference curve.
- **Change robot dimensions**: `config/robot.yaml`, plus the constants at the top of `arena_env.py` (`R_ROBOT`, `HEAD_R`, ...), `ROBOT_RADIUS` in `tools/layout.py`; rebuild the layout pools.
- **Change layouts** (obstacle count/size, arena size): `tools/layout.py`; delete `rl/pool_*.npz`; `python3 tests/test_layout.py`.
- **Train a new model**: `.venv/bin/python rl/train.py run11 8e6`; the GUI lists any `runs/**/*.npz`; copy a good one to `weights/`.
- **Tune the safety layer**: constants at the top of `rl/shield.py` (`HORIZON`, `MARGIN`); evaluate with the 2D CLI both with and without it.

## Pitfalls (each cost time once)
- Never run `pkill -f 'gz sim'` inside a command whose text contains `gz sim`: it kills your own shell. Use `tools/killsim.sh`.
- Run one simulator at a time; stale `gz sim` processes silently corrupt tests. Check with `pgrep -af "[g]z sim"`.
- The simulator is paused and stepped manually; do not press play in the Gazebo GUI.
- `runs/gazebo_results.json` is overwritten by every Gazebo evaluation.
- Wheel link roll must stay -1.5707963 (otherwise the robot drives backwards while odometry says forward); centre of mass x = -0.03 keeps it from tipping.
- Decoding AprilTags needs the multi-scale search in `vision.py`; a single scale misses some ids.
- Host/container: edit `~/.bashrc` only with container-guarded blocks (it is shared with the Fedora host); use `python3 -m venv --system-site-packages` (PEP 668).
- Ignore the separate ULTRA_ROBOTS workspace; packages here use the `snr_` prefix, not `arena_*`.

## Conventions
Every module starts with a docstring (purpose, usage). Constants are named and commented where they are defined. Layout seeds 0-199 are
evaluation-only. Commits are local only; tag milestones (current: `v0.3-rl-policy`).
