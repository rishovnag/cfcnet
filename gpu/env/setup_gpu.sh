#!/usr/bin/env bash
# One-time environment set-up on a CUDA GPU node (Linux x86_64; developed on an NVIDIA H100).
#   bash gpu/env/setup_gpu.sh && source .venv/bin/activate && python gpu/check_gpu.py
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r gpu/env/requirements-gpu.txt
pip install -e .
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
