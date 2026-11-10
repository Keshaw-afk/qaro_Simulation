# Benchmark protocol: PPO vs Q-learning (tabular, DQN) vs deterministic controllers

Written and committed **before any benchmark result was produced** (branch `benchmark-policies`, 2026-11-10).
Results go to `benchmark/RESULTS.md`, whatever they are. Any later deviation from this protocol is listed there.

## Question
How does the deployed policy (`weights/final_policy.npz`, PPO) compare with (a) classic tabular Q-learning,
(b) deep Q-learning (DQN), (c) a deterministic controller using the same sensors, and (d) the existing DWA planner
that sees the true map (privileged reference)?

## Contenders
| Id | Method | Information used | Trained? |
|---|---|---|---|
| `ppo` | `weights/final_policy.npz` (run10 checkpoint at 7M steps) | 59-number observation | yes (existing, not retrained) |
| `ppo_final` | `runs/run10/final.npz` (same run, 8M steps, never selected on evaluation layouts) | same | yes (existing) |
| `tabq` | Tabular Q-learning on a discretised observation | derived from the same 59 numbers | yes, here |
| `dqn` | DQN (Stable-Baselines3), MLP 59-256-256-45 tanh, same as PPO's actor | same 59 numbers | yes, here (Colab T4) |
| `rule` | Hand-written reactive controller (no learning) | same 59 numbers | no (parameters tuned on validation) |
| `dwa` | `rl/expert.py` dynamic-window planner | **true positions, true obstacle map** (privileged) | no |
| `goto` | Go-to-goal baseline (`rl/evaluate.py`), ignores everything | goal only | no |

All use the same 45 discrete (speed, turn-rate) commands (`rl/actions.py`), the same 2D environment
(`rl/arena_env.py`), and the same safety layer (`rl/shield.py`) unless the condition says it is off.
The safety layer is part of the environment for every learner (it swaps the proposed action inside `step_async`),
exactly as during PPO training: learners store the action they proposed.

## Data split
- **Test: evaluation layouts 100-199** (seeds 100-199, `rl/pool_eval.npz` index = seed). No method was trained, tuned
  or checkpoint-selected on them. (PPO's checkpoint was selected on layouts 0-99, so 0-99 are not a fair test for it.)
- Layouts 0-99 are also reported for reference only (PPO's selection set).
- **Validation (all tuning and checkpoint selection of the new methods): training seeds 2900-2999**, held out from
  their training. New learners train on seeds 1000-2899. (PPO trained on 1000-2999; the 100-layout difference is negligible.)

## Protocol
`mode="eval"` of `ArenaVecEnv`: fixed starts/goals, 4 robots per layout, robots stop at the goal (0.12 m) or on
collision, timeout 900 steps (90 s), no domain randomisation. Deterministic controllers/greedy policies.

## Metrics
Per robot: success, collision (by cause: obstacle, wall, robot), timeout; mean time to goal over successes;
all-four-success per layout; safety-layer override rate.
**Uncertainty:** 95% bootstrap confidence intervals resampling **layouts** (10,000 resamples), because the 4 robots of
one layout are correlated. Differences to `ppo` use a **paired** bootstrap over the same layouts.

## Conditions
1. Safety layer **on** (deployment configuration, primary result).
2. Safety layer **off** (how much each method relies on it).

## Training budgets and fairness
- `dqn`, `tabq`: final training 8M environment steps (PPO's budget), **3 seeds each**. Report every seed, the mean,
  and the seed chosen by validation (never by test). Checkpoints every 1M steps; the best on validation is kept
  (the same selection rule PPO used: success - 0.5 x collision, but on validation layouts).
- Tuning budget: `dqn` up to 6 configurations x 1M steps; `tabq` up to 6 configurations x 2M steps; `rule` up to 50
  parameter sets. All judged on validation only. PPO's tuning history (10 runs, see docs/DOCUMENTATION.md section 7)
  is larger, which favours PPO; this is stated with the results.
- `tabq`: report number of states and the fraction visited.
- Compute and wall time per method reported.

## Gazebo confirmation
PPO and the best one or two rivals on **layouts 100-119** in Gazebo (full camera/lidar pipeline), safety layer on.
PPO is re-run on these same layouts (its old 0-19 result is not reused).

## Reproducibility
Code in `benchmark/`; trained rival weights in `runs/benchmark/` (not committed, AGENTS.md rule 6); raw per-robot
outcome arrays and summary in `benchmark/results/*.json` (committed).

## Amendments (appended; the text above is unchanged)
1. **2026-11-10 ~12:45, tabq state representation.** The coarse 6,480-state table plateaued at ~60% validation success
   with 99% of states visited, i.e. limited by the representation rather than by data or bugs. To avoid a strawman, a
   finer representation (157,464 states, see `tabq.py`) is tuned on validation (configs `fa`, `fd`, `fg`, 2M steps)
   and, if it beats the coarse table on validation, trained for 8M steps x 3 seeds. Both variants are reported on the
   test set. Decided before any tabq test-set evaluation.
2. **2026-11-10 ~13:12, outcome of amendment 1.** Fine tabq at 2M steps on validation: fa 24.0% success / 28.5% coll,
   fd 20.3% / 37.0%, fg 28.3% / 31.3%, all far below coarse config a (64.5% / 24.8%). By the amendment-1 rule the fine
   variant is **not** a contender; the coarse table is the tabq result. Because the fine table had visited only ~50% of
   its states and was still improving (data-starved, not representation-limited), one **exploratory** run (fg, seed 1,
   8M steps) is made and reported separately, labelled exploratory, never chosen over coarse by test results.
3. **2026-11-10 ~13:38, rule grid extension.** The 36-set grid's best (84.0% / 1.5% validation) lay on the lower edge
   of horizon, margin and w_head. Before any rule test evaluation, the grid is extended past that edge
   (`rule.PARAM_GRID_EXT`: horizon 0.7, margin 0.01, w_head 0.1, 14 new sets, total 50 = the pre-registered budget).
   The choice remains the best validation score over all 50 sets. The 36-set choice is kept in
   `results/rule_best_params_grid36.json`.
