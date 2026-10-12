#!/usr/bin/env bash
# Start / stop / query the NVIDIA Multi-Process Service (lets many worker processes share the GPU's SMs).
#   bash gpu/mps.sh start | stop | status
# Needs permission to run nvidia-cuda-mps-control (root or the GPU's owner on most cloud images).
set -uo pipefail
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}
case "${1:-status}" in
  start)
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then echo "MPS: already running"; exit 0; fi
    nvidia-cuda-mps-control -d && echo "MPS: started (pipe $CUDA_MPS_PIPE_DIRECTORY)" ;;
  stop)
    echo quit | nvidia-cuda-mps-control && echo "MPS: stopped" ;;
  status)
    if echo get_server_list | nvidia-cuda-mps-control 2>/dev/null; then echo "MPS: running"; else echo "MPS: not running"; fi ;;
  *) echo "usage: $0 start|stop|status"; exit 2 ;;
esac
