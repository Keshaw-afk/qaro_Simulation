#!/usr/bin/env bash
# Final tabular Q runs (benchmark/PROTOCOL.md): chosen config x 3 seeds x 8M steps, in parallel (~0.5 GB RAM each).
# Run inside the ubuntu distrobox. Usage: bash benchmark/run_tabq_final.sh CONFIG
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for s in 1 2 3; do
    .venv/bin/python benchmark/tabq.py train "$1" $s 8e6 final_s$s > runs/benchmark/logs/tabq_final_s$s.log 2>&1 &
done
wait
echo "tabq final done"
