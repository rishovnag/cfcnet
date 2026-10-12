# Data

All series are daily index sessions from **Yahoo Finance**, downloaded with `yfinance`. The CSV files
are **not** redistributed in this repository. Recreate them with the download script, and compare
their checksums with `checksums.sha256` to confirm that your copy matches the one used in the paper.
Readers who need bit-identical inputs can obtain the exact snapshots from the authors, or from the
release asset `cfcnet-data-snapshots.tar.gz` where the vendor's terms permit attaching it.

```bash
pip install yfinance
python data/download_data.py --end 2026-10-01   # extended samples, frozen at the paper's end date
python data/download_data.py --primary          # the NIFTY 100 primary sample (to 2024-07-16)
sha256sum -c data/checksums.sha256              # run from inside data/  (macOS: shasum -a 256 -c)
```

## Files

| File | Index | Yahoo ticker | Used in |
|---|---|---|---|
| `raw/NIFTY50.csv` | NIFTY 50 | `^NSEI` | cross-market |
| `raw/NIFTY100.csv` | NIFTY 100 | `^CNX100` | extended evaluation, walk-forward, history ablation |
| `raw/NIFTY500.csv` | NIFTY 500 | `^CRSLDX` | split experiment |
| `raw/SPX.csv` | S&P 500 | `^GSPC` | cross-market |
| `raw/N225.csv` | Nikkei 225 | `^N225` | cross-market |
| `raw/SSE.csv` | SSE Composite | `000001.SS` | cross-market |
| `primary/NIFTY100_2024-07-16.csv` | NIFTY 100 | `^CNX100` | headline comparison and robustness (Sections 6.3-6.4) |

`manifest.json` records, for every file in `raw/`, the download time (UTC), the `yfinance`
version, the first and last session, the row count and the SHA-256 checksum. The extended samples
of the paper run from the oldest session Yahoo provides to 30 September 2026.

Schema: `Date, Open, High, Low, Close, Adj Close, Volume`. Prices are not adjusted by `yfinance`
(`auto_adjust=False`); the vendor's `Adj Close` column is kept, and for price indices it equals the
close (the diagnostics check this). `src/cfcnet/data.py::load_index` drops sessions with a missing
field, a non-positive price or a duplicate date. Zero volumes are used as reported.

Vendors revise history, so a later download can differ from the files used in the paper; the
checksums tell you whether it does.

## Sample start (genuine OHLC)

In the early history of some series the vendor records a single daily price (O = H = L = C): the
S&P 500 before the 1960s and through 1971, and the Nikkei 225 before April 1988. With
`start: auto_ohlc` (the default in `configs/paper.yaml`), each sample starts at the first session
after the last trailing one-year block in which more than 5% of sessions are flat; isolated flat
sessions later in the sample are kept. The same start applies to both targets.
`scripts/01_diagnostics.py` reports the vendor start, the analysed start and the number of
excluded and zero-volume sessions in `results/tables/dataset_diagnostics.csv`.
