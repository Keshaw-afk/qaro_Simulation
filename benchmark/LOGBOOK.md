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
