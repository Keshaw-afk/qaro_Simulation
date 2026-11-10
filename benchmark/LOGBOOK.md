# Benchmark logbook (lab notebook)

Chronological record of what was done, what went wrong, and why decisions were made. Times are local (IST), 2026-11-10.
Machine-readable measurements of every job: `results/measurements.jsonl`. Protocol: `PROTOCOL.md`. Results: `RESULTS.md`.

## Setup
- **~10:55** Branch `benchmark-policies` from `main` (eea05de). Hardware branch parked (tag `parked/hardware-experimentation-2026-11-10`).
- Resources: laptop i5-12450H (8 cores / 12 threads, 15 GB RAM, owner asked to keep >= 3-4 GB free) inside the
  `ubuntu` distrobox (Python 3.12, numpy 1.26, SB3 2.9.0, torch CPU); Google Colab with a Tesla T4 (2 vCPU, 12.7 GB RAM,
  Python 3.13, numpy 2.4, torch 2.14 cu130). Plan: CPU-bound work (tabular Q, rule controller, evaluations, Gazebo)
  locally; DQN training (neural network) on the T4.
- **11:10** Protocol committed before any result (ea39d35).

## Findings during setup
- **PPO checkpoint selection leak (known before running):** `rl/train.py` picks `best.npz` by evaluating on evaluation
  layouts 0-99, so 0-99 are not a clean test for PPO. Test set moved to layouts 100-199.
- **Harness reproduces PPO exactly** on 0-99: 91.5% success / 4.0% collision / 4.5% timeout (as in docs), locally and on
  Colab (numpy 2.4); observation-parity test passes on Colab.
- **Safety layer swaps PPO's action on ~20% of active robot-steps** (0-99), ~26% on 100-199. Higher than expected.
- **2D env throughput:** ~9,100 robot-steps/s single process (28 ms per 256-robot step) locally.
- **DQN on laptop CPU:** 321 steps/s (smoke test, while 6 other jobs ran) -> 8M steps would take ~7 h: confirms GPU use.

## Bugs and mistakes (all caught before any reported number)
1. **tabq terminal-state bug (logic).** After an episode ended, the final (terminal) state was also used as the robot's
   current state for the next update, although the robot had already been respawned / given a new goal. Affected
   ~0.6% of updates. First 6-config tuning sweep discarded (`runs/benchmark/discarded/tabq_buggy_terminal_state`),
   fixed (`s_boot` vs `s_next`), rerun. Found by line-by-line review after the owner asked about logic bugs.
2. **Meter double-counted guest CPU time** in the machine-wide busy time (/proc/stat). Fixed.
3. **Operational: `pgrep -f PATTERN | xargs kill` killed my own shell** (exit 144) because the pattern was in the command
   text, exactly the pitfall in AGENTS.md rule 2. Processes are now killed by PID only (or with a `[b]racket` pattern).
4. A per-layout hashing job was slow because `np.load(npz)[key][i]` decompresses the whole array on every access; load
   arrays once.

## Training-pool discovery (affects the original project)
- Colab rebuilt `rl/pool_train.npz` from the current code and **385 / 2000 training layouts differed** from the local
  file (evaluation pool identical). Cause: the local pool was built at 23:45 on 2026-10-18 with `tools/layout.py` from
  commit 06418bd; 15 minutes later commit 6bae2a8 added the goal-visibility rule (goals must be visible to the overhead
  camera). The pool was never rebuilt. **PPO was trained on layouts from the older generator** (19% of its training
  layouts would be rejected today). In the 2D env (no camera) this is a minor shift in layout distribution.
- Decision: every new learner trains on exactly PPO's pool. Colab rebuilds it with the 06418bd generator; array hash
  verified identical to the local file (d7bbd86c...).

## Energy measurement
- Intel RAPL counters are root-only on Fedora (since CVE-2020-8694, "PLATYPUS" power side channel). Owner chose not to
  open them. CPU energy is therefore an **estimate**: CPU seconds x 45 W / 12 threads (i5-12450H), basis stored with
  each record. GPU energy on Colab is measured (NVML counter).

## Logging (owner: "log and measure everything")
- Per job: wall/CPU time, estimated CPU energy, GPU energy/power/utilisation/memory, throughput, CPU clock, temperature,
  power source, min free RAM, load average, command line, versions, git commit.
- Per evaluation: outcome, collision cause, time, path length and efficiency, speed, time stopped, closest approach to
  obstacles/robots, near misses, steering flips, safety-layer swaps per robot, proposed/executed action histograms,
  decision latency (batched and single-robot), model size.
- Training: validation every 1M steps (`log.csv`), tabq curve every 100k (`train_curve.csv`), DQN SB3 log (`sb3/progress.csv`).

## Early numbers (first, interrupted baseline evaluation; re-run with full metrics)
- PPO on clean test 100-199: 90.2% success, 4.5% collision (vs 91.5% on its selection set 0-99): selection optimism ~1.3 points.
- PPO without the safety layer: 68.0% success, 25.2% robot-robot collisions (test) - the policy depends heavily on it.
- PPO behaviour (8 test layouts): ~226 steering-direction flips per minute, 96% path efficiency, 0.18 m/s mean speed.

## Tuning results
- **tabq** (6 configs x 2M steps, seed 0, validation seeds 2900-2999; score = success - 0.5 collision):
  a 64.5% / 24.8% coll (score 0.521, **chosen**), d 63.0% / 24.0% (0.510), f 55.3% / 29.5%, c 54.0% / 29.5%,
  b 51.0% / 39.3%, e 32.8% / 52.3%. Visit-count learning rates beat constant ones; gamma 0.95 hurt. 97% of the
  6,480 states visited by 2M steps. 2,300-2,700 steps/s per process (4-6 processes in parallel), 12-15 min per run.
  Final runs: config a, seeds 1-3, 8M steps each (started ~12:15).
- **dqn** (Colab T4, 1M steps each, 2 processes in parallel): ~1,650 steps/s per process; GPU only 4-7% busy - the
  bottleneck is Colab's 2 vCPUs (simulator + SB3 replay bookkeeping), not the network. a 63.0% / 4.0% coll / 33.0% timeout;
  b (lr 3e-4) 26.0% / 0.3% / 73.8%; **c (4 gradient steps) 63.5% / 0.5% / 36.0% (score 0.633, chosen)**;
  d (exploration 0.5) 62.3% / 3.3%; e (target update 2k) 54.7% / 4.8%; f (ReLU) 54.7% / 6.8%. DQN collides far less
  than tabq but times out a lot at 1M steps. Tuning runs used the code before the extra logging (logging only).
  Final runs: config c, seeds 1-3, 8M steps, all three concurrently on Colab (started ~12:50 local; code hashes verified).

## Baseline evaluation (full metrics, ~12:10-12:45)
- Test 100-199, shield on / off: PPO 90.2% / 68.0%; PPO final (8M, never selected) 89.5% / 65.0% (dSucc vs PPO
  -0.8 [-3.0, +1.8]: checkpoint selection did not matter); go-to-goal 54.2% / 42.0%; DWA (privileged) 79.8% / 79.5%.
- **DWA without the shield beats PPO without the shield by +11.5 points [+4.2, +18.5]** and has 0 collisions; its
  failures are 20% timeouts. PPO relies on the safety layer for robot-robot avoidance.
- **Cross-check:** go-to-goal on 0-99 without shield = 43.2% / 56.8% exactly as docs/DOCUMENTATION.md; with shield
  55.8% vs documented 55.7%. DWA on 0-99 with shield 79.5% vs documented "about 78%": not reconciled; the documented
  number predates the env robot-clearance bug fix (docs section 7), probable but unverified cause.
- Deployment cost: one robot's decision PPO 113 us (92,717 params, 372 KB), go-to-goal 22 us; DWA has no single-robot
  path (needs the simulator state); its batched evaluation is 2-3x slower than PPO's.
- Gazebo adapter smoke test (PPO, layout 100, loaded machine): 3 success + 1 timeout, 194 s wall; real Gazebo runs are
  scheduled after the CPU-heavy jobs (AGENTS.md: clock stalls under load).
- ~12:58 Owner reconnected the Colab tab: same runtime (uptime unchanged), the three DQN runs kept going. Google Drive
  mounted (owner authorised); `MyDrive/qaro_benchmark/` receives checkpoints (.npz), logs, SB3 progress and
  measurements every 5 min. A crash would lose the replay buffer, so a run could not be resumed exactly, only its best
  checkpoint so far kept.

## Plateau handling (~12:45)
- Coarse tabq plateaued at ~60% validation success / ~27% collisions by 4M steps, 99% of 6,480 states visited; training
  success with eps 0.05 (~62%) agrees with validation, TD error still falling (0.50 -> 0.37): no sign of a bug, the
  representation is the limit. Protocol amendment 1: finer 157,464-state variant (configs fa/fd/fg), decided before any
  tabq test evaluation. tabq update rewritten with np.unique (sparse; identical math, tested) because the dense
  bincount over 7M table entries per step would be slow.
- Metric note: `stopped_frac` over all robots is dominated by robots that time out (they sit or spin for up to 900
  steps: PPO 5.5% on 8 layouts without timeouts vs 30.6% over the test set); added `stopped_frac_success`.

## Fine tabq outcome (~13:12)
- 2M steps, validation: fa 24.0% / 28.5% coll, fd 20.3% / 37.0%, fg 28.3% / 31.3% (coarse a: 64.5% / 24.8%). ~78k of
  157k states visited (50%), still rising; ~1,900 steps/s per process. The finer table learns far more slowly: each
  state is seen ~24x less often. Fine variant fails the amendment-1 rule; coarse stays the tabq contender.
- Exploratory fg seed 1 to 8M started (protocol amendment 2) to see whether more data closes the gap.
- Coarse finals at 7M: s1 64.8% / 19.8%, s2 63.8% / 22.3%, s3 62.8% / 20.0% (validation): flat at ~60-65%.

## tabq final results (~13:20)
- Finals (config a, 8M): best validation checkpoints s1 64.8% (7M), s2 64.8% (6M), **s3 71.5% (8M, chosen by validation)**.
- **Test 100-199, shield on:** s1 64.5%, s2 61.0%, **s3 69.5%** [64.2-74.8] (coll 20.0%, all obstacle/wall; dSucc vs PPO
  -20.8 [-25.5, -16.0]); mean of seeds 65.0%. Shield off: 53.8 / 51.7 / 52.5%, collisions 32-42% (robot 14-24%).
  The table's robot avoidance comes almost entirely from the safety layer, like PPO's; obstacle collisions (~19%) are
  what it cannot learn with 3 coarse lidar regions. Overrides 6-13% (PPO 26%): it is more timid near robots.
- Evaluation is fast (~1-2 min per condition per seed, 3 in parallel).
- 13:19 Security housekeeping (not benchmark): host shell stopped exporting the GitHub token (distrobox put it on a
  world-readable `podman exec` command line). No effect on any experiment.

## Unattended pipeline and how to resume (~13:30)
The owner may close the Claude session; the rest runs without it.
- **Laptop:** `benchmark/run_remaining.sh` (started detached with `setsid nohup` inside the distrobox; log
  `runs/benchmark/logs/run_remaining.log`). Order: wait for rule tuning (re-runs it if its process died) -> 2D test
  evaluation of `rule` -> 2D evaluation of the exploratory fine tabq -> Gazebo PPO (100-119) -> wait up to 6 h for
  the DQN weights -> 2D evaluation of each DQN seed, Gazebo of the validation-chosen seed -> Gazebo of the better of
  rule / tabq_s3 (test success). Every step is skipped if its output exists, so re-running the script is safe.
- **Colab:** `/content/package_when_done.sh` waits for the 3 DQN runs, then writes `MyDrive/qaro_benchmark/dqn_final.tar.gz`
  (+ `.sha256`, `DQN_FINISHED.txt`). The Colab tab must stay open until then.
- **Manual step if nobody fetches it:** download `dqn_final.tar.gz` from Drive, check the sha256, then
  `tar xzf dqn_final.tar.gz -C ~/robotics/swarm_nav_rl` (it contains `runs/benchmark/dqn/final_s{1,2,3}/` with `DONE` markers, plus `colab_extra/` with the Colab
  measurements and logs, kept apart so nothing local is overwritten). The script picks the weights up within 2 min.
- Still to do by hand afterwards: `benchmark/RESULTS.md`, docs overview, commit.

## Rule controller tuning (~13:34)
- 36 sets, validation, 2 processes, 92 min (~5 min per set per process). Best: horizon 1.0, margin 0.02, w_clear 2.0,
  w_head 0.3 -> **84.0% success / 1.5% collision / 14.5% timeout** (all collisions obstacle, none robot), mean time
  9.9 s. The top ~6 sets are within 1.7 points of each other. Strongest rival so far on validation (PPO is not
  measured on these layouts, so no comparison yet).
- The best sat on the lower edge of 3 parameters: pipeline stopped before the rule test evaluation (none run),
  grid extended by 14 sets (amendment 3), 7 processes; the pipeline restarts automatically afterwards.

## Pipeline bugs, Colab reset, DQN results (~15:05)
- **My pipeline script had two bugs** (no result affected, ~70 min lost): the Python path was relative and broke
  inside `cd benchmark`, so the rule and fine-tabq evaluations failed instantly; and `set -u` made ROS's setup.bash
  abort, so Gazebo never started. The script then waited for DQN. Fixed (absolute path, `set +u` around `source`),
  checked, relaunched at ~15:00.
- Rule grid extension finished: best of 50 sets is horizon 0.7, margin 0.02, w_clear 2.0, w_head 0.1:
  **87.3% / 2.3% / 10.5%** on validation (was 84.0% / 1.5%). Edge again on w_head and horizon, but the budget
  (50 sets) is spent; stated as a limitation.
- Exploratory fine tabq at 8M: 53.5% / 25.3% on validation (coarse finals 61-72%). More data did not close the gap.
- **Colab runtime reset** (uptime 3 min at 15:01 IST); last Drive backup 14:23 IST at 6.14M steps. Bundle rebuilt from
  Drive, downloaded to ~/Downloads (sha256 4e1ccb25...), installed in `runs/benchmark/dqn/final_s*` with DONE markers
  saying "partial". Colab measurements in `results/measurements_colab.jsonl` (the final runs never wrote a Meter
  record; their time/throughput come from `log.csv` / `sb3/progress.csv`: ~955 steps/s per run, 6,300 s to 6M).
- **DQN validation (best checkpoint):** s1 89.5% / 0.5% (6M), **s2 91.75% / 1.0% (5M, chosen)**, s3 87.0% / 0.0% (6M).
  DQN was still noisy between checkpoints (s1 dipped to 55% at 3M, s2 to 54.5% at 2M).

## Decision: DQN stays at 6M; test evaluations (~15:20)
- Owner decided: no 8M rerun; DQN is evaluated as it stands (amendment 4). DQN 2D evaluations started alongside the
  PPO Gazebo run (load average ~2, so the Gazebo clock is not at risk).
- **Rule controller, test 100-199:** shield on 84.0% [79.8-88.0] / 2.5% coll / 13.5% timeout, dSucc vs PPO
  **-6.2 [-10.2, -2.2]**; shield off **84.5% / 0.5% coll**, vs PPO without shield **+16.5 [+9.5, +23.2]**. All four
  robots succeed in 56-57% of layouts; mean time to goal 9.6 s. It needs no safety layer: robot-robot avoidance is
  built in (camera slots), whereas PPO and tabq learned to lean on the safety layer.
- **Exploratory fine tabq, test:** 54.2% / 26.0% with shield, 46.2% / 39.0% without.
