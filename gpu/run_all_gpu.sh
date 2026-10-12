#!/usr/bin/env bash
# Full pipeline on a CUDA GPU node (developed on an NVIDIA H100): every table and figure of the paper from the downloaded data.
# Usage:  bash gpu/run_all_gpu.sh [PAPER_DIR]
#   PAPER_DIR  optional manuscript folder; data_facts.tex, table_*.tex and fig_*.pdf are written there
# Runtime settings (GPUs, workers per GPU, seeds per job, MPS) come from configs/gpu.yaml;
# run gpu/autotune.py --write once to tune them for the machine.
set -euo pipefail
cd "$(dirname "$0")/.."
PAPER=""; [[ $# -ge 1 ]] && PAPER="--paper-dir $1"

python gpu/check_gpu.py
python scripts/01_diagnostics.py $PAPER
python scripts/09_data_figures.py $PAPER
python gpu/plan.py
mkdir -p runs/logs

# CPU-bound benchmarks (ARIMA) run alongside the GPU training on the cores the workers leave free
W=$(python -c "import yaml;r=yaml.safe_load(open('configs/gpu.yaml'));print(len(r['gpus'])*r['workers_per_gpu']*r['cpu_threads_per_worker'])")
FREE=$(( $(nproc) - W )); [[ $FREE -lt 1 ]] && FREE=1
python scripts/03_baselines.py --jobs "$FREE" > runs/logs/baselines.log 2>&1 &
BASE_PID=$!

python gpu/run_parallel.py           # all network training; merges the CSV logs at the end
wait $BASE_PID

python scripts/10_primary_window.py      # headline window and robustness (240 small runs, CPU)
python scripts/12_latency.py --device gpu  # serial GPU latency (supplementary; runs/latency_gpu.csv)
python scripts/12_latency.py --device cpu --repeats 30   # the serial CPU benchmark reported in the paper
python scripts/06_dm_tests.py
python scripts/11_skill_decomposition.py
python scripts/07_make_tables.py $PAPER
python scripts/08_make_figures.py $PAPER
