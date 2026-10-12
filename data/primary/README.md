# NIFTY 100 primary sample

`NIFTY100_2024-07-16.csv` is the NIFTY 100 series (Yahoo Finance `^CNX100`) from 30 November 2005 to
16 July 2024: 4,580 sessions after `load_index` removes the rows with missing fields. It is the
sample of the headline comparison and of its robustness analysis (Sections 6.3-6.4 of the paper;
`scripts/10_primary_window.py`). Its 95-5 test window runs from 3 August 2023 to 16 July 2024.

SHA-256: `6285a080e91eec6ba545efd367c6d5b2ec53683f94f8cccd71e1101d1ee2c45f`

Obtain it with `python data/download_data.py --primary` (which reports whether the checksum
matches) or, bit-identical, from the authors or the release asset `cfcnet-data-snapshots.tar.gz` (see `data/README.md`).
