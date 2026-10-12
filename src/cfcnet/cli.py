"""Helpers shared by the scripts in scripts/."""
from __future__ import annotations

import argparse

from .io import load_config


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default=None, help="YAML config (default: configs/paper.yaml)")
    ap.add_argument("--data-dir", default=None, help="folder with the index files (default: data/raw)")
    ap.add_argument("--out", default=None, help="run folder (default: runs/)")
    ap.add_argument("--epochs", type=int, default=None, help="override protocol.epochs (e.g. 2 for a smoke test)")
    ap.add_argument("--jit", action="store_true", help="XLA-compile CfC/LTC (GPU)")
    return ap


def load(args) -> dict:
    cfg = load_config(args.config)
    if getattr(args, "epochs", None):
        cfg["protocol"]["epochs"] = args.epochs
    if getattr(args, "jit", False):
        cfg["protocol"]["jit_compile_ncps"] = True
    return cfg


def motor_for(target: str) -> int:
    return 4 if target == "ohlc" else 1


def units_for(cfg: dict, target: str, width: str = "default") -> int:
    return int(cfg["capacity"][target][width])


def add_grid_args(ap):
    ap.add_argument("--targets", nargs="*", choices=["adjclose", "ohlc"], help="override protocol.targets")
    ap.add_argument("--reprs", nargs="*", choices=["levels", "returns"], help="override protocol.representations")
    return ap


def grid(cfg: dict, args):
    """(target, representation) pairs to run."""
    ts = getattr(args, "targets", None) or cfg["protocol"]["targets"]
    rs = getattr(args, "reprs", None) or cfg["protocol"]["representations"]
    return [(t, r) for t in ts for r in rs]


def parse_seeds(s: str | None, default: int) -> list[int]:
    """'30' -> seeds 0..29 (a count) ; '0,3,7' -> [0,3,7] ; '5-9' -> 5..9 ; '0-0' -> [0]"""
    if s is None:
        return list(range(default))
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    if int(s) < 1:
        raise ValueError("a single number is a seed COUNT; use '0-0' or '0,' for seed 0 alone")
    return list(range(int(s)))
