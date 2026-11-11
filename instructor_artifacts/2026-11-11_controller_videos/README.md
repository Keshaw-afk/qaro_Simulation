# Controller videos on the designed arenas (2026-11-11)

Three videos, one per controller, each driving the **five designed crossing arenas 9001-9005** (`docs/DESIGNED_LAYOUTS.md`)
in Gazebo (full pipeline: overhead camera, lidar, safety layer on; stepped simulation recorded from the angled demo camera).

| File | Controller | Length |
|---|---|---|
| `ppo_designed_arenas.mp4` | PPO (deployed policy, `weights/final_policy.npz`) | 1:35 |
| `dqn_designed_arenas.mp4` | DQN seed 2 (`runs/benchmark/dqn/final_s2/best.npz`) | 1:51 |
| `rule_designed_arenas.mp4` | Rule controller, no learning (`benchmark/rule.py`) | 1:27 |

## Results (robots 1-4: red, orange, blue, yellow)

| Arena | PPO | DQN | Rule controller |
|---|---|---|---|
| 9001 crossroads | ok, ok, ok, ok | ok, ok, ok, ok | ok, ok, ok, ok |
| 9002 roundabout | ok, ok, collision, ok | ok, ok, ok, ok | timeout, timeout, timeout, timeout |
| 9003 single door | timeout, ok, ok, timeout | timeout, timeout, collision, timeout | collision, timeout, timeout, collision |
| 9004 pillar forest | ok, ok, collision, collision | ok, timeout, timeout, collision | collision, timeout, ok, ok |
| 9005 corridor swap | ok, ok, ok, ok | ok, timeout, timeout, ok | ok, ok, ok, ok |
| **Robots home** | **15 / 20** | **11 / 20** | **10 / 20** |

One episode per arena, so this shows behaviour, not a ranking (for that see `benchmark/RESULTS.md`). These arenas are
deliberately hard and unlike the training layouts.

## Remake them
```
bash tools/make_comparison_videos.sh            # from the repository root; about 10 minutes; -> runs/demo/compare/final/
```
Needs the Gazebo half of the setup (`docs/SETUP.md` section 0) and ffmpeg. Gazebo runs vary a little between runs, so
per-robot results can differ from the table above by a robot or two. The `.mp4` files are not in git (`.gitignore`).

## Editing
Each arena: a 1 s title card, about 0.5 s of stillness, the run, about 1 s holding the final positions. Stretches where
nothing moves for more than 1.5 s are shortened to 0.8 s. Runs longer than 20 s play normally for 12 s, then at
**6x speed** (labelled on screen) until their last 2 s. The overlay clock always shows true simulation time, so every cut
or speed-up is visible (for example, the rule controller's roundabout run: robots stand still for most of 70 s and shrink
to about 12 s). Raw recordings: `runs/demo/compare/` (not in git).
