import numpy as np
import pandas as pd

from demand.quantiles import BAND_DAYS, MIN_BAND_DAYS, QUANTILES, offsets


def residuals(n, seed=0):
    days = pd.date_range('2026-01-01', periods=n)
    return pd.DataFrame(np.random.default_rng(seed).normal(0, 100, (n, 96)), index=days)


def test_needs_enough_history():
    r = residuals(MIN_BAND_DAYS + 1)
    assert offsets(r, r.index[-1]) is None                 # only days up to D-2 count
    assert offsets(r, r.index[-1] + pd.Timedelta('1D')) is not None


def test_uses_only_days_up_to_d_minus_2():
    r = residuals(200)
    day = r.index[150]
    poisoned = r.copy()
    poisoned.loc[day - pd.Timedelta('1D'):] = 1e9           # D-1 onwards must not leak in
    assert np.allclose(offsets(r, day), offsets(poisoned, day))
    old = r.copy()
    old.loc[:day - pd.Timedelta(f'{BAND_DAYS + 3}D')] = 1e9  # nor anything older than the window
    assert np.allclose(offsets(r, day), offsets(old, day))


def test_quantiles_ordered_and_calibrated():
    r = residuals(200)
    off = offsets(r, r.index[-1])
    assert off.shape == (len(QUANTILES), 96)
    assert (np.diff(off, axis=0) > 0).all()
    assert abs(off[-1].mean() - 100 * 1.645) < 15           # q95 of N(0, 100)
