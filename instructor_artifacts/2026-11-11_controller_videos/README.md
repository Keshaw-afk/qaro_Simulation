# Controller videos (2026-11-11)

Three videos, one per controller, each driving the **same five arenas** in Gazebo (full pipeline: overhead camera,
lidar, safety layer on; stepped simulation recorded from the angled demo camera):

| File | Controller | Length |
|---|---|---|
| `ppo_5_arenas.mp4` | PPO (deployed policy, `weights/final_policy.npz`) | 1:37 |
| `dqn_5_arenas.mp4` | DQN seed 2 (`runs/benchmark/dqn/final_s2/best.npz`) | 1:54 |
| `rule_5_arenas.mp4` | Rule controller, no learning (`benchmark/rule.py`) | 1:29 |

## Arenas and how they were chosen
Not hand-picked: three test layouts were drawn at random (Python `random.Random(20261111).sample`) from the 58 held-out
test layouts 100-199 whose starts and goals are all visible from the video camera (`tools/demo_layouts.py se --all`), and
one designed arena was drawn from 9001-9005. Designed arena 9001 was added afterwards on request.

1. Test layout 126  2. Test layout 146  3. Test layout 197  4. Designed arena 9001, crossroads  5. Designed arena 9003, single door
(designed arenas: `docs/DESIGNED_LAYOUTS.md`)

## Results (robots 1-4: red, orange, blue, yellow)

| Arena | PPO | DQN | Rule controller |
|---|---|---|---|
| 126 | collision, ok, ok, ok | ok, ok, ok, ok | ok, ok, ok, ok |
| 146 | ok, ok, ok, ok | ok, ok, ok, ok | ok, ok, ok, ok |
| 197 | ok, ok, ok, ok | ok, ok, ok, ok | ok, ok, ok, ok |
| 9001 crossroads | ok, ok, ok, ok | ok, ok, ok, ok | ok, ok, ok, ok |
| 9003 single door | timeout, ok, ok, timeout | timeout, timeout, collision, timeout | collision, timeout, timeout, collision |

One episode per arena, so this shows behaviour, not a ranking (for that see `benchmark/RESULTS.md`).

## Editing
Each arena: a title card (1.2 s), about 0.5 s of stillness, the run, about 1 s holding the final positions. Stretches
where nothing moves for more than 1.5 s are shortened to 0.8 s. Runs longer than 30 s play normally for 20 s, then at
**4x speed** (labelled on screen) until their last 3 s. The overlay clock always shows true simulation time, so every
cut or speed-up is visible. Raw recordings: `runs/demo/compare/` (not in git).
