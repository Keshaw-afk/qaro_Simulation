#!/usr/bin/env bash
# DQN finals on Colab: chosen config c, seeds 1-3, 8M steps (stopped at 6.14M by a runtime reset, PROTOCOL amendment 4),
# all three concurrently (~2.7 GB RAM each). Back up runs/benchmark/dqn to Drive regularly: a reset loses the replay buffers.
# (Copied into the repo after the run; the commands are the ones that ran.)
set -euo pipefail
cd "$(dirname "$0")/.."
export SNR_TORCH_THREADS=1
for s in 1 2 3; do python benchmark/dqn.py c $s 8e6 final_s$s cuda > runs/benchmark/logs/dqn_final_s$s.log 2>&1 & done
wait; echo done > runs/benchmark/logs/dqn_final.done
