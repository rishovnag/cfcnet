"""Out-of-envelope deployment guard (Section 6.5 / Conclusions of the paper).

The CFC closed-form update has no saturating envelope on its latent trajectory, so a
deployment should flag (and optionally clip) forecasts that leave a plausibility band
derived from the training window. Because the band is computed from the training targets
only, the guard uses no future information.
"""
from __future__ import annotations

import numpy as np


def plausibility_band(train_y_scaled: np.ndarray, margin: float = 0.5):
    """Band [min - margin*range, max + margin*range] per target column (scaled units)."""
    y = train_y_scaled.reshape(-1, train_y_scaled.shape[-1])
    lo, hi = y.min(axis=0), y.max(axis=0)
    rng = hi - lo
    return lo - margin * rng, hi + margin * rng


def apply_guard(pred_scaled: np.ndarray, band, fallback: np.ndarray | None = None):
    """Return (guarded forecasts, boolean flags per session).

    Sessions whose forecast leaves the band in any column are flagged. If `fallback`
    (e.g. the random-walk forecast y_t) is given, flagged sessions are replaced by it;
    otherwise the forecast is clipped to the band."""
    lo, hi = band
    flags = ((pred_scaled < lo) | (pred_scaled > hi)).any(axis=1)
    out = pred_scaled.copy()
    if fallback is not None:
        out[flags] = fallback[flags]
    else:
        out = np.clip(out, lo, hi)
    return out, flags
