# AGENTS.md: instructions for AI coding agents (and a fast orientation for humans)

Read this first. Then `docs/DOCUMENTATION.md` (what the system is and why) and `docs/CODE_GUIDE.md` (how the code is organised).
Claude Code reads `CLAUDE.md`, which points here.

## What this project is
Four differential-drive robots (TortoiseBot-style) drive to per-robot goals in a random-obstacle arena in Gazebo Harmonic + ROS 2 Jazzy.
One shared RL policy (PPO, 45 discrete actions) controls all four from each robot's own sensors; a rule-based safety layer (`rl/shield.py`)
vetoes robot-robot collision courses. State: finished and evaluated (about 91% success in Gazebo, see `docs/DOCUMENTATION.md`).

## Environment you need
Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic (gz sim 8.x). Setup recipe: `docs/SETUP.md`. Verify with:
```
source /opt/ros/jazzy/setup.bash && python3 tools/check_env.py
```
Two Python environments exist on purpose:
- **System `python3` + ROS sourced**: Gazebo evaluation, the desktop app, tools, layout tests (needs the Gazebo Python bindings).
- **`.venv` (created with `--system-site-packages`, `pip install -r requirements.txt`)**: training, 2D evaluation, observation-parity test (needs torch, stable-baselines3, gymnasium).
If `.venv` does not exist, only the second group is unavailable; do not install heavy packages into the system Python.

## Common commands (also available as `make <target>`, see `make help`)
| Task | Command |
|---|---|
| Fast tests (no simulator) | `make test` |
| Desktop app | `python3 swarm_gui.py` |
| Gazebo evaluation, standard arena, layouts 1-3 | `source /opt/ros/jazzy/setup.bash; SNR_FIRST=1 python3 rl/eval_gazebo.py weights/final_policy.npz 4 0` |
| Same in the big 8 x 6 m arena | prefix `SNR_ARENA=big` (layout seeds 51-100 were used for the reported big-arena results) |
| Watch a layout live in the Gazebo window | prefix `SNR_LIVE=1` |
| 2D evaluation (seconds) | `.venv/bin/python rl/eval2d.py weights/final_policy.npz 100` |
| Train | `.venv/bin/python rl/train.py RUN 8e6` (about 2 h on the development machine; writes `runs/RUN/`) |
Argument conventions: `eval_gazebo.py WEIGHTS N_END [N_VIDEO_END]` runs layouts `SNR_FIRST .. N_END-1`. Full variable list in its docstring.

## Rules that prevent the known failure modes
1. **Run one simulator per `GZ_PARTITION`.** Simulators in the same partition share topics and silently corrupt each other's results. Use a different `GZ_PARTITION` for concurrent runs (the app does this automatically). Check with `pgrep -af "[g]z sim"`.
2. **Never put `pkill -f 'gz sim'` (or any `pkill -f PATTERN`) in a command whose own text contains PATTERN**: it kills your own shell. Use `tools/killsim.sh` (kills ALL simulators, including other people's runs: avoid when others are running) or kill by PID.
3. **Headless evaluation is lockstep**: the world is paused and stepped (`GzSim.step`); do not press play or call unpause. Live mode (`SNR_LIVE=1`) is the exception: it plays in real time (`GzSim.play`).
4. **Observation, action table and layout generator are shared contracts** between the 2D env, deployment and Gazebo: change them in one place only (`rl/actions.py`, `OBS` code in `rl/arena_env.py` + `rl/deploy.py`, `tools/layout.py`) and run `tests/test_obs_parity.py`. Changing them invalidates trained weights.
5. **Evaluation layouts are seeds 0-199; training uses seeds >= 1000.** Never train on, or tune against, evaluation seeds without saying so.
6. **Do not commit** `runs/`, `rl/pool_*.npz`, `.venv/`, videos or screenshots (see `.gitignore`). The only tracked weights are `weights/final_policy.npz`.
7. Report results honestly: quote counts (e.g. 73/80 robots) and what was not diagnosed; do not round up. `runs/gazebo_results.json` is overwritten by every run, so copy results you want to keep (the app writes to `runs/gui/`).
8. Gazebo runs are slow (about 1 minute per layout, 2-3 minutes in the big arena): run long jobs in the background and poll sparingly.

## Definition of done for a change
1. `make test` passes (layout, safety layer; plus observation parity if `.venv` exists).
2. If behaviour of the policy path changed: 2D evaluation on 100 layouts before/after, and a Gazebo run of at least a few layouts.
3. Docs updated (`docs/DOCUMENTATION.md` for behaviour, `docs/CODE_GUIDE.md` for structure, `CHANGELOG.md`), docstrings kept current.
4. Small commits, imperative message. Do not push, force-push or rewrite history unless the owner asks.

## Where things are
`swarm_gui.py` desktop app | `rl/` env, training, deployment, safety layer, evaluators | `tools/` world/robot generators, Gazebo wrapper, vision, checks |
`config/robot.yaml` robot parameters (SOURCED / DERIVED / ESTIMATED) | `weights/` final policy | `docs/` documentation and recorded results | `tests/` tests.
