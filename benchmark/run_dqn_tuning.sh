#!/usr/bin/env bash
# DQN tuning on Colab (T4 GPU, 2 vCPU): configs a-f, 1M steps each, 2 at a time. Run from the repo root on Colab
# after rebuilding rl/pool_train.npz with the 06418bd generator (see LOGBOOK "Training-pool discovery").
# (Copied into the repo after the run; the commands are the ones that ran.)
set -euo pipefail
cd "$(dirname "$0")/.."
export SNR_TORCH_THREADS=1
printf '%s\n' a b c d e f | xargs -P 2 -I{} sh -c 'python benchmark/dqn.py {} 0 1e6 tune_{} cuda > runs/benchmark/logs/dqn_tune_{}.log 2>&1'
echo done > runs/benchmark/logs/dqn_tuning.done
