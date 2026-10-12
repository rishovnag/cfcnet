# Optional GPU backend

The reference pipeline in `scripts/` trains every run with Keras `model.fit`, one model per process,
and runs on any CPU (with LSTM optionally on an Apple-silicon GPU). This folder adds an optional
backend for data-centre GPUs that trains the same models, on the same windows, with the same
protocol, many times faster. It was developed on an NVIDIA H100 (80 GB) and works on any recent
CUDA GPU. `tests/test_fast.py` checks that it reproduces the reference path.

Use it only to re-run the training grid. Everything else — diagnostics, benchmarks, statistics,
tables and figures — is the same code as in `scripts/`.

## Set-up

```bash
bash gpu/env/setup_gpu.sh            # venv, tensorflow[and-cuda], ncps, pip install -e .; prints nvidia-smi
source .venv/bin/activate
python gpu/check_gpu.py              # versions, CUDA/cuDNN, MPS, and fast path == model.fit on the GPU
python gpu/autotune.py --write       # optional (~15 min): best workers_per_gpu and pack for this host
```

## Running the whole paper

```bash
python data/download_data.py --end 2026-10-01 && python data/download_data.py --primary
bash gpu/run_all_gpu.sh <paper-folder>       # PAPER folder optional; outputs also go to results/
```

`run_all_gpu.sh` runs the diagnostics and data figures, builds the job plan, trains the full grid
on the GPU(s) while the ARIMA benchmarks run on the free CPU cores, runs the primary-window
experiment (small, on the CPU), the latency benchmarks, the Diebold–Mariano tests, the skill
decomposition, and writes every table and figure. It is resume-safe.

## What is optimised

| Level | Technique | Where |
|---|---|---|
| Kernel | All 400 epochs run inside **one compiled `tf.function`**; CfC and LTC are **XLA-compiled**, removing per-step Python and launch overhead. LSTM and GRU stay on the fused **cuDNN** kernel, in one graph-mode loop. | `src/cfcnet/fast.py` |
| Model | **Seed packing**: `pack` seeds of one cell train in one graph. Parameters are disjoint, the loss is the sum of per-model MSEs and Adam is element-wise, so each model receives exactly its own update (verified against `model.fit`). | `fast.train_pack` |
| Process | **Many workers per GPU** (`workers_per_gpu`, default 16) sharing the GPU through **NVIDIA MPS**, so the small recurrent kernels of different runs overlap. Each worker has its own GPU memory cap, pinned CPU cores and thread budget. | `gpu/run_parallel.py`, `src/cfcnet/gpuenv.py` |
| Node | Any number of GPUs (`--gpus 0 1 2 3`), round-robin. Workers on several hosts can share one `runs/` folder on a network file system. | `gpu/run_parallel.py` |
| Queue | The grid is a plan of independent jobs (`runs/jobs.jsonl`) ordered **longest first**. Jobs are claimed with `O_EXCL` lock files and each run is written as an **atomic row file**; locks of killed workers are detected through their pid, logged in `runs/failed/` and taken over. | `src/cfcnet/jobs.py`, `src/cfcnet/worker.py` |
| Compile | A **persistent XLA cache** shared by all workers (`runs/xla_cache`). | `gpuenv.worker_env` |

Numerical settings default to fidelity with the reference path: TF32 off, fp32 throughout.
`configs/gpu.yaml` holds every runtime setting (`gpus`, `workers_per_gpu`, `pack`, `mps`,
`memory_limit_mb`, `tf32`, …); command-line flags override it. The experiment itself is defined
only in `configs/paper.yaml`.

## Step by step

```bash
python gpu/plan.py                                     # the full grid: fixed-origin, walk-forward, history ablation
python gpu/plan.py --only fixed --experiment headline --targets ohlc --reprs levels   # a subset
python gpu/plan.py --import <runs dir>                 # reuse runs already finished elsewhere (e.g. on a CPU machine)
python gpu/run_parallel.py                             # all GPUs/workers from configs/gpu.yaml
python gpu/run_parallel.py --gpus 0 1 --workers-per-gpu 12
python gpu/status.py                                   # progress, held locks, failures, last log line per worker
python gpu/merge.py                                    # row files -> runs/fixed_origin.csv, walk_forward.csv, history_ablation.csv
```

The launcher prints throughput and an ETA every minute, restarts crashed workers and merges the row
files when the queue is empty. Smoke test without a GPU:

```bash
python gpu/plan.py --out runs_smoke --experiment headline --targets ohlc --reprs levels --arch cfc gru --seeds 2
python gpu/run_parallel.py --plan runs_smoke/jobs.jsonl --out runs_smoke --cpu-only --workers-per-gpu 2 --epochs 2
```

## Timings and latency

Training and test times recorded during a parallel GPU run are contended (several runs share the
GPU) and are flagged `timing_contended`. The inference latencies reported in the paper come from the
serial CPU benchmark `scripts/12_latency.py` (`runs/latency_cpu.csv`). `scripts/12_latency.py --device gpu`
benchmarks each architecture alone on the GPU and writes `runs/latency_gpu.csv` as a supplementary
measurement.

## MPS

MPS is started by the launcher (`gpu/mps.sh start|stop|status`). It needs permission to run
`nvidia-cuda-mps-control` and, on shared clusters, the GPU in default (not exclusive-process)
compute mode or MPS enabled by the scheduler (e.g. Slurm `--gres=mps`). Without MPS, use `--no-mps`
with fewer workers per GPU (4–6).
