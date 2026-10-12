#!/usr/bin/env bash
# Full pipeline: every table and figure of the paper from raw data.
# Usage:  bash scripts/run_all.sh [PAPER_DIR] [--jit] [--workers=N] [--threads=N] [--gpu-workers=N]
#   PAPER_DIR    optional manuscript folder; data_facts.tex, table_*.tex and fig_*.pdf are written there
#   --jit        XLA-compile CfC/LTC on a CUDA GPU (with --workers=1 --gpu-workers=0)
#   --workers=N  parallel CPU runs (default: all cores); --threads=N  threads per run (default 1)
#   --gpu-workers=N  runs on the GPU, LSTM only (default: 8 if TensorFlow sees a GPU, else 0)
# All training scripts are resume-safe; re-running skips finished runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PAPER=""; EXTRA=""; PAR=""
for a in "$@"; do
  case "$a" in
    --workers=*|--threads=*) PAR="$PAR $a" ;;
    --*) EXTRA="$EXTRA $a" ;;
    *) PAPER="--paper-dir $a" ;;
  esac
done

python scripts/01_diagnostics.py $PAPER
python scripts/09_data_figures.py $PAPER
# one pool for all four experiments, so no cores sit idle between them
python scripts/02_fixed_origin.py --experiment headline capacity split_sweep cross_market $EXTRA $PAR
python scripts/04_walk_forward.py $EXTRA $PAR
python scripts/05_history_ablation.py $EXTRA $PAR
python scripts/10_primary_window.py $EXTRA $PAR
python scripts/03_baselines.py $PAR
python scripts/06_dm_tests.py
python scripts/11_skill_decomposition.py
python scripts/12_latency.py --device cpu --repeats 30   # serial, on an otherwise idle machine
python scripts/07_make_tables.py $PAPER
python scripts/08_make_figures.py $PAPER
