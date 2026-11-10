#!/usr/bin/env bash
# Tabular Q tuning sweep (benchmark/PROTOCOL.md): 6 configurations x 2M steps, at most 4 processes at once
# (each ~0.5 GB RAM; keeps >= 4 GB free on the 15 GB development laptop). Run inside the ubuntu distrobox from the
# repository root. Logs: runs/benchmark/logs/tabq_tune_*.log
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf '%s\n' a b c d e f | xargs -P "${SNR_PAR:-4}" -I{} sh -c '.venv/bin/python benchmark/tabq.py train {} 0 2e6 tune_{} > runs/benchmark/logs/tabq_tune_{}.log 2>&1'
echo "tabq tuning done"
