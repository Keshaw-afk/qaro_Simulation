# Benchmark results: PPO vs Q-learning (tabular, DQN) vs deterministic controllers

Branch `benchmark-policies`, 2026-11-10. Protocol (written before any result): `PROTOCOL.md`, with 4 amendments.
Day-by-day record, bugs and decisions: `LOGBOOK.md`. Every number below is regenerated from raw per-robot files by
`make_results.py` (output `results/summary.json`); illustrated PDF: `REPORT.pdf`; raw files are in `results/raw/` and `results/gazebo__*.json`.

## Short answer
- **With the safety layer (the deployed configuration), PPO is not beaten, but it is matched.** DQN (validation-chosen
  seed) is statistically level with PPO in the 2D simulator (90.0% vs 90.2%, difference -0.2 [-4.0, +3.5]) **and in
  Gazebo** (73/80 robots each), with fewer collisions (2D 1.5% vs 4.5%; Gazebo 2 vs 5 robots) but slower arrivals
  (16 s vs 11 s) and more timeouts. DQN got 75% of PPO's training budget and much less tuning. DQN's seeds vary:
  83-90% on test.
- **Without the safety layer, PPO is the weakest learned method after tabular Q.** The hand-written rule controller
  (a dynamic-window planner on the same 59 inputs) reaches 84.5% with 0.5% collisions vs PPO's 68.0% with 29.2%
  (+16.5 [+9.5, +23.2]). PPO and DQN learned to rely on the safety layer for robot-robot avoidance (PPO: 25.2% of
  robots hit another robot without it; the layer replaces PPO's action on 26% of steps).
- **The rule controller's 2D advantage did not carry over to Gazebo:** 80.0% (64/80) vs PPO 91.2%, difference -11.2
  [-23.8, 0.0], mostly timeouts (13 robots). Hand-built path prediction suffered more from real physics and sensing
  than the learned policies did. (20 layouts: wide interval.)
- **Tabular Q-learning is far behind** (69.5% chosen seed, 20% collisions): limited by its state compression; a finer
  table learned more slowly and did worse (54.2%).
- The privileged DWA planner (sees the true map) gets 79.8% with almost no collisions but 20% timeouts.

## Contenders
| Id | What it is | Inputs | Training / tuning here |
|---|---|---|---|
| PPO | `weights/final_policy.npz`, run10 at 7M steps (existing) | 59 numbers | none (10 earlier runs of tuning history) |
| PPO 8M | same run, final checkpoint (never selected on evaluation layouts) | same | none |
| DQN | SB3 DQN, same network shape as PPO's actor (59-256-256-45 tanh) | same | 6 configs x 1M; finals 3 seeds, **6M steps** (amendment 4) on Colab T4 |
| Tabular Q | Q-table on a 6,480-state discretisation | derived from same | 6 configs x 2M; finals 3 seeds x 8M |
| Tabular Q fine | 157,464-state table (amendment 1, exploratory) | same | 3 configs x 2M; 1 seed x 8M |
| Rule | dynamic-window search over the 45 commands, lidar points + camera robot slots (`rule.py`) | same | 50 parameter sets (amendment 3) |
| DWA | `rl/expert.py` planner | **true map and positions** | none |
| Go-to-goal | drive at the goal | goal only | none |

All use the same 45 commands, the same 2D environment and the same safety layer (`rl/shield.py`) unless "off".
Selection of seeds and checkpoints used validation layouts (training seeds 2900-2999) only, never the test set.

## 2D simulator, test layouts 100-199 (400 robots), safety layer ON
95% intervals bootstrap over layouts; differences paired with PPO on the same layouts.

| Method | Success % [95% CI] | Collision % | Timeout % | All four % | Time to goal s | Overrides % | dSuccess vs PPO [95% CI] |
|---|---|---|---|---|---|---|---|
| PPO (deployed) | 90.2 [86.5, 93.5] | 4.5 | 5.2 | 71.0 | 10.9 | 25.9 | - |
| PPO, 8M checkpoint | 89.5 [85.8, 92.8] | 4.0 | 6.5 | 69.0 | 10.7 | 24.5 | -0.8 [-3.0, +1.8] |
| DQN seed 1 | 88.0 [84.5, 91.3] | 0.5 | 11.5 | 64.0 | 15.4 | 6.7 | -2.2 [-6.0, +1.5] |
| **DQN seed 2 (chosen)** | 90.0 [86.8, 93.0] | 1.5 | 8.5 | 68.0 | 16.1 | 8.0 | -0.2 [-4.0, +3.5] |
| DQN seed 3 | 83.0 [78.8, 87.0] | 0.5 | 16.5 | 51.0 | 15.4 | 9.1 | -7.2 [-11.0, -3.5] |
| **Rule controller** | 84.0 [79.8, 88.0] | 2.5 | 13.5 | 56.0 | 9.6 | 9.8 | -6.2 [-10.2, -2.2] |
| Tabular Q seed 1 | 64.5 [59.0, 70.0] | 20.2 | 15.2 | 22.0 | 14.7 | 6.1 | -25.8 [-31.0, -20.2] |
| Tabular Q seed 2 | 61.0 [55.8, 66.0] | 27.5 | 11.5 | 17.0 | 13.8 | 6.5 | -29.2 [-34.2, -24.5] |
| **Tabular Q seed 3 (chosen)** | 69.5 [64.2, 74.8] | 20.0 | 10.5 | 31.0 | 15.9 | 12.8 | -20.8 [-25.5, -16.0] |
| Tabular Q fine (exploratory) | 54.2 [48.2, 60.2] | 26.0 | 19.8 | 18.0 | 16.4 | 4.7 | -36.0 [-41.8, -30.2] |
| DWA (privileged: true map) | 79.8 [75.0, 84.2] | 0.2 | 20.0 | 47.0 | 11.9 | 2.4 | -10.5 [-15.0, -6.0] |
| Go-to-goal | 54.2 [48.5, 60.2] | 41.5 | 4.2 | 20.0 | 8.9 | 42.1 | -36.0 [-41.5, -30.5] |

Mean over seeds: DQN 87.0%, tabular Q 65.0%.

## 2D simulator, test layouts 100-199, safety layer OFF
| Method | Success % [95% CI] | Collision % (robot-robot) | Timeout % | All four % | Time to goal s | dSuccess vs PPO (also off) [95% CI] |
|---|---|---|---|---|---|---|
| PPO (deployed) | 68.0 [62.3, 73.8] | 29.2 (25.2) | 2.8 | 39.0 | 10.3 | - |
| PPO, 8M checkpoint | 65.0 [59.0, 71.0] | 32.5 (29.2) | 2.5 | 36.0 | 10.0 | -3.0 [-7.0, +0.8] |
| DQN seed 1 | 75.8 [70.5, 80.8] | 12.5 (12.5) | 11.8 | 47.0 | 14.6 | +7.8 [+1.5, +13.8] |
| **DQN seed 2 (chosen)** | 76.0 [71.0, 81.0] | 16.5 (15.8) | 7.5 | 47.0 | 15.7 | +8.0 [+1.8, +14.2] |
| DQN seed 3 | 74.8 [69.5, 80.0] | 12.5 (12.5) | 12.8 | 43.0 | 15.3 | +6.8 [-0.8, +14.2] |
| **Rule controller** | 84.5 [80.0, 88.5] | 0.5 (0.0) | 15.0 | 57.0 | 9.6 | +16.5 [+9.5, +23.2] |
| Tabular Q seed 1 | 53.8 [47.8, 59.8] | 31.8 (14.0) | 14.5 | 15.0 | 13.8 | -14.2 [-21.2, -7.2] |
| Tabular Q seed 2 | 51.7 [46.5, 57.0] | 38.2 (16.8) | 10.0 | 10.0 | 13.0 | -16.2 [-23.8, -9.0] |
| **Tabular Q seed 3 (chosen)** | 52.5 [46.5, 58.2] | 41.8 (23.5) | 5.8 | 17.0 | 13.8 | -15.5 [-23.0, -8.0] |
| Tabular Q fine (exploratory) | 46.2 [40.8, 51.7] | 39.0 (16.2) | 14.8 | 7.0 | 15.2 | -21.8 [-29.2, -14.5] |
| DWA (privileged: true map) | 79.5 [74.5, 84.0] | 0.0 (0.0) | 20.5 | 48.0 | 11.9 | +11.5 [+4.2, +18.5] |
| Go-to-goal | 42.0 [36.8, 47.5] | 58.0 (19.0) | 0.0 | 9.0 | 8.1 | -26.0 [-32.2, -19.8] |

Reference only, layouts 0-99 (PPO's checkpoint-selection set, so optimistic for PPO), success on / off: PPO 91.5 /
71.5, DQN s2 86.8 / 76.8, rule 82.2 / 82.8, tabular Q s3 67.8 / 51.2, DWA 79.5 / 80.0, go-to-goal 55.8 / 43.2 (all
rows in `make_results.py` output). PPO's selection optimism: 91.5 on 0-99 vs 90.2 on 100-199.

## Gazebo, layouts 100-119 (80 robots), full pipeline, safety layer on
Overhead camera + AprilTags, 360-beam lidar, odometry, physics; `rl/eval_gazebo.py` unchanged (episode limit 70 s,
shorter than the 2D 90 s). PPO, the validation-chosen DQN seed, and the better of rule / tabular Q on the 2D test.

| Method | Success % [95% CI] | Robots ok / coll / timeout | All four % | Time to goal s | dSuccess vs PPO [95% CI] | Decision us (median / p99) |
|---|---|---|---|---|---|---|
| PPO (deployed) | 91.2 [85.0, 97.5] | 73 / 5 / 2 | 70.0 | 10.4 | - | 98 / 342 |
| **DQN seed 2** | 91.2 [86.2, 96.2] | 73 / 2 / 5 | 65.0 | 15.5 | +0.0 [-5.0, +5.0] | 84 / 233 |
| **Rule controller** | 80.0 [70.0, 88.8] | 64 / 3 / 13 | 45.0 | 9.4 | -11.2 [-23.8, +0.0] | 1247 / 4518 |

PPO's 91.2% agrees with its earlier Gazebo record (89.8% on layouts 1-50), a check that the benchmark harness drives
Gazebo correctly. The rule controller's decision time in Gazebo is for 4 robots at once on a loaded machine (2D
single-robot: 160 us); both are far below the 100 ms control period.

## How they behave (2D test, safety layer on)
| Method | Path efficiency | Mean speed m/s | Stopped (successes) | Steering flips /min | Median closest obstacle m | Median closest robot m | Obstacle near-miss steps /robot |
|---|---|---|---|---|---|---|---|
| PPO | 0.967 | 0.127 | 4.4% | 218 | 0.166 | 0.150 | 1.2 |
| DQN s2 | 0.961 | 0.090 | 9.6% | 249 | 0.170 | 0.193 | 1.3 |
| Rule | 0.988 | 0.085 | 0.9% | 12 | 0.117 | 0.227 | 117 |
| Tabular Q s3 | 0.857 | 0.087 | 12.1% | 204 | 0.134 | 0.196 | 12.5 |
| DWA | 0.932 | 0.066 | 0.2% | 4 | 0.087 | 0.172 | 46.9 |

- PPO is the fastest and most direct learned driver, and passes closest to other robots (it lets the safety layer
  sort out conflicts). DQN keeps more distance from robots, drives slower and stops more.
- Both learned policies flip steering direction 200-250 times per minute (jittery commands); the planners 4-12.
- The rule controller skims obstacles (margin 2 cm chosen by tuning; 117 near-miss steps per robot): fine with the 2D
  simulator's exact geometry, a plausible reason it lost ground in Gazebo, where sensing and motion are imperfect.

## Cost
| | Parameters | Model size | One decision (laptop) |
|---|---|---|---|
| PPO / DQN (same network) | 92,717 | 372 KB | 113 / 83 us |
| Tabular Q (coarse / fine) | 291,600 / 7.1M entries | 1.2 MB / 28 MB | ~140 us / 84 us |
| Rule | 7 numbers | - | 160 us |
| Go-to-goal | - | - | 22 us |

Training and tuning (CPU energy is an **estimate**, CPU seconds x 3.75 W per busy thread, because the Intel RAPL
counters are root-only; GPU energy measured by NVML):
| | Where | Wall time | Compute | Energy |
|---|---|---|---|---|
| Tabular Q (13 runs incl. tuning, finals, fine) | laptop, 1 thread each | 6.4 h summed (runs in parallel) | 6.3 CPU h | ~0.024 kWh CPU (est.) |
| Rule tuning (50 sets) | laptop, 2-7 processes | 1.7 h | 4.3 CPU h | ~0.016 kWh (est.) |
| DQN tuning (6 x 1M) | Colab T4 + 2 vCPU | 1.0 h summed | GPU 4-7% busy | 0.029 kWh GPU (measured) |
| DQN finals (3 x 6M, concurrent) | Colab T4 + 2 vCPU | 1.75 h each, ~955 steps/s each | GPU 8-15% busy, ~30 W board | ~0.05 kWh GPU (est. from power; runs did not finish their meter) |
| Gazebo, 20 layouts | laptop | 18 min (PPO) - 31 min (rule) | | |
| 2D evaluation, 100 layouts | laptop | 1-5 min per condition | | |

PPO's own training (run10, 8M steps) took about 2 h on the laptop (docs/DOCUMENTATION.md); its energy was not measured.
The Colab GPU was mostly idle: a 93k-parameter network is tiny for a T4, and Colab's 2 CPU cores (the simulator and
replay buffer) were the bottleneck. Colab's real contribution was extra RAM and parallel capacity.

## Deviations from the protocol (all recorded in PROTOCOL.md amendments and LOGBOOK.md)
1. Fine tabular Q representation tried after the coarse table plateaued (amendment 1); it lost on validation, so the
   coarse table is the tabular Q result; one exploratory 8M run reported separately (amendment 2).
2. Rule grid extended past its edge within the pre-registered 50-set budget (amendment 3). The final choice still
   sits on the edge of two parameters (horizon 0.7, w_head 0.1); a wider search might improve it slightly.
3. **DQN trained 6M steps, not 8M** (amendment 4): the Colab runtime was reset; the replay buffers were lost; the
   owner chose to evaluate the 6M checkpoints. Two of three seeds were still improving at 6M.
4. DQN seed 3 was evaluated twice (manual run and the pipeline overlapped); the evaluation is deterministic and the
   second run overwrote identical results.

## Limitations
- **Tuning effort favours PPO**: it is the result of 10 training runs and reward/environment fixes (docs section 7);
  DQN had 6 short configurations, tabular Q 9, the rule controller 50 cheap parameter sets.
- **Seeds**: PPO is a single trained policy (no seed spread available); DQN's spread (83-90%) suggests PPO's single
  number may also be a favourable draw.
- **Gazebo uses 20 layouts** (80 robots): intervals are wide (+-6 to 9 points); only large differences are resolved.
- **Training pool**: PPO and every new learner trained on the same stale pool (built before the goal-visibility rule,
  LOGBOOK "Training-pool discovery"); the comparison is fair, but all methods saw slightly different layouts than
  today's generator produces.
- The 2D simulator's motion model is exact; planners that predict motion (rule, DWA) benefit from that in 2D.
- CPU energy is an estimate; per-job CPU time and wall time are measured.

## Reproduce
Everything the numbers depend on is in the repository: the raw per-robot results (`results/`), the trained models and
their training logs (`runs/benchmark/{dqn,tabq}/*/best.npz`, `log.csv`, force-added although `runs/` is ignored), the
rule controller's tuning (`runs/benchmark/rule_tuning.json`, `results/rule_best_params.json`) and both layout pools.
**Never delete or rebuild `rl/pool_train.npz`:** it was built with an older layout generator (LOGBOOK.md); a rebuilt
pool differs on 385 of 2000 layouts, which silently changes the validation layouts and any retraining.
Run from the repository root unless a line says `cd benchmark`. Environments as in `AGENTS.md` (`.venv` for 2D and
training, system `python3` with ROS sourced for Gazebo).

**Level 1: recompute every table and the PDF from the committed raw files (about 1 minute)**
```
.venv/bin/python benchmark/make_results.py          # all tables above, rewrites results/summary.json (should not change)
python3 benchmark/make_report.py                    # REPORT.pdf; plain Python 3 (no packages) + google-chrome or chromium on PATH
.venv/bin/python benchmark/test_benchmark.py        # harness tests; add --slow for the PPO regression (91.5/4.0/4.5)
```

**Level 2: re-evaluate the committed models (2D: about 2-8 minutes per model; Gazebo: about 20-30 minutes per model)**
```
cd benchmark && ../.venv/bin/python evaluate_all.py ppo=npz:../weights/final_policy.npz \
    dqn_s2=npz:../runs/benchmark/dqn/final_s2/best.npz rule=rule tabq_s3=tabq:../runs/benchmark/tabq/final_s3/best.npz dwa=dwa goto=goto
# -> results/raw/NAME__{test,ref0-99}__shield{1,0}.json (deterministic: identical files)
source /opt/ros/jazzy/setup.bash
GZ_PARTITION=bench_ppo .venv/bin/python benchmark/eval_gazebo_bench.py ppo npz:weights/final_policy.npz 100 120
GZ_PARTITION=bench_dqn .venv/bin/python benchmark/eval_gazebo_bench.py dqn_s2 npz:runs/benchmark/dqn/final_s2/best.npz 100 120
GZ_PARTITION=bench_rule .venv/bin/python benchmark/eval_gazebo_bench.py rule rule 100 120
cd benchmark && ../.venv/bin/python eval_designed.py   # designed crossing layouts 9001-9005 (docs/DESIGNED_LAYOUTS.md)
```
2D results are deterministic for a given software build. Exact environment of the published numbers: Ubuntu 24.04, Python
3.12.3, Ubuntu's numpy 1.26.4 and scipy 1.11.4 (apt, venv with `--system-site-packages`), torch 2.14.1 (CPU),
stable-baselines3 2.9.0, gymnasium 1.4.0. A pip-only environment (pip's numpy wheel with its own BLAS, scipy 1.17.1,
torch 2.15.0, sb3 2.9.2; checked 2026-11-11 on all 100 test layouts, safety layer on) gave identical outcomes for PPO and
the rule controller on all 400 robots, and for DQN on 399 of 400 (success 89.75% instead of 90.0%): floating-point
rounding differs between BLAS builds and occasionally flips a close decision. Designed layouts in that environment: same
outcomes, DQN finish times shifted by a few steps on one layout. Gazebo results vary by a few robots between runs (real-time sensor timing), so expect
the counts to differ slightly, not exactly.

**Level 3: retrain (hours; DQN needs a GPU)**
```
bash benchmark/run_tabq_tuning.sh && bash benchmark/run_tabq_final.sh a   # tabular Q (CPU, about 6 h)
bash benchmark/run_dqn_tuning.sh && bash benchmark/run_dqn_final.sh       # DQN (GPU; run on Colab: about 1.7 h per seed on a T4)
.venv/bin/python benchmark/rule.py tune 2 && .venv/bin/python benchmark/rule.py tune_ext 7   # rule controller (about 2 h)
bash benchmark/run_remaining.sh                     # evaluations + Gazebo, unattended (needs ROS)
```
Retrained models will not match bit for bit (GPU nondeterminism, thread scheduling); the protocol (PROTOCOL.md) fixes the
seeds, budgets and selection rules, so results should land within the reported intervals. The DQN finals were stopped
at 6M of the planned 8M steps (amendment 4); change `8e6` to `6e6` in `run_dqn_final.sh` to reproduce that.
