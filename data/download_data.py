#!/usr/bin/env python
"""Download the latest daily OHLCV history of every index from Yahoo Finance via yfinance,
starting at the OLDEST session Yahoo provides (period="max").

    pip install yfinance
    python data/download_data.py                # writes data/raw/*.csv, data/manifest.json, data/checksums.sha256
    python data/download_data.py --end 2026-10-01   # freeze the sample end (exclusive), as for the paper
    python data/download_data.py --primary          # the NIFTY 100 primary sample (to 2024-07-16) of Sections 6.3-6.4

Tickers are read from configs/paper.yaml (indices.<name>.ticker). The script records the exact
download time, the yfinance version, and the first/last session, row count and SHA-256 of every
file in data/manifest.json, so a reader can verify that their copy matches the paper's.
Prices are NOT adjusted by yfinance (auto_adjust=False); the vendor's "Adj Close" column is kept.

--primary writes data/primary/NIFTY100_2024-07-16.csv and compares its SHA-256 with the snapshot used
in the paper (data/primary/README.md). Vendors revise history, so a fresh download can differ; the
exact snapshot is available from the authors (see data/README.md).

After downloading, regenerate the data facts used by the manuscript:
    python scripts/01_diagnostics.py --paper-dir <paper folder>
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(ticker: str, end: str | None, retries: int = 3):
    import pandas as pd
    import yfinance as yf
    for k in range(retries):
        try:
            kw = {"period": "max"} if end is None else {"start": "1900-01-01", "end": end}
            df = yf.download(ticker, interval="1d", auto_adjust=False, actions=False, progress=False,
                             threads=False, **kw)
            if len(df):
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index()
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
                if "Adj Close" not in df:
                    df["Adj Close"] = df["Close"]
                return df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        except Exception as exc:  # network hiccups, rate limits
            print(f"  {ticker}: attempt {k + 1} failed: {exc}")
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"could not download {ticker}")


PRIMARY = {"ticker": "^CNX100", "end": "2024-07-17", "file": "NIFTY100_2024-07-16.csv",
           "sha256": "6285a080e91eec6ba545efd367c6d5b2ec53683f94f8cccd71e1101d1ee2c45f"}


def download_primary() -> None:
    out = REPO / "data" / "primary"
    out.mkdir(parents=True, exist_ok=True)
    df = fetch(PRIMARY["ticker"], PRIMARY["end"])
    path = out / PRIMARY["file"]
    df.to_csv(path, index=False)
    got = sha256(path)
    print(f"NIFTY100 {PRIMARY['ticker']} {len(df)} rows  {df.Date.iloc[0]} -> {df.Date.iloc[-1]}")
    if got == PRIMARY["sha256"]:
        print("checksum matches the snapshot used in the paper")
    else:
        print("WARNING: checksum differs from the snapshot used in the paper (vendor revision);\n"
              "         use the exact snapshot (see data/README.md) to reproduce Tables 4 and 8 exactly.")


def main():
    import yaml
    import yfinance as yf
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "data" / "raw"))
    ap.add_argument("--end", default=None, help="exclusive last date YYYY-MM-DD (default: latest available)")
    ap.add_argument("--primary", action="store_true", help="download the NIFTY 100 primary sample only")
    args = ap.parse_args()
    if args.primary:
        download_primary()
        return
    cfg = yaml.safe_load(open(REPO / "configs" / "paper.yaml"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "Yahoo Finance via yfinance", "yfinance_version": yf.__version__,
                "downloaded_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "period": "max" if args.end is None else f"oldest available to {args.end} (exclusive)",
                "auto_adjust": False, "files": {}}
    for name, icfg in cfg["indices"].items():
        df = fetch(icfg["ticker"], args.end)
        path = out / icfg["file"]
        df.to_csv(path, index=False)
        manifest["files"][icfg["file"]] = {"index": name, "ticker": icfg["ticker"], "rows": int(len(df)),
                                           "first": df.Date.iloc[0], "last": df.Date.iloc[-1], "sha256": sha256(path)}
        print(f"{name:9s} {icfg['ticker']:10s} {len(df):6d} rows  {df.Date.iloc[0]} -> {df.Date.iloc[-1]}")
    (REPO / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (REPO / "data" / "checksums.sha256").write_text(
        "".join(f"{v['sha256']}  raw/{k}\n" for k, v in manifest["files"].items())
        + f"{PRIMARY['sha256']}  primary/{PRIMARY['file']}\n")
    print("wrote data/manifest.json and data/checksums.sha256")


if __name__ == "__main__":
    main()
