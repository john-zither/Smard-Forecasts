"""Long-format features for a standalone forecast, one row per (day, region, quarter-hour).

SMARD's own forecast is never an input: it is only the benchmark. The
target is actual load over a recent level (the mean load of the 28 days
ending D-2), and every load feature is scaled the same way, so one model
serves regions whose load differs eightfold. A feature for delivery day D
only uses what is published by 12:00 on D-1 (see demand.baselines): full
days up to D-2, D-1 up to 10:15, and the weather forecast made two days
earlier.
"""

import numpy as np
import pandas as pd

from demand.baselines import MORNING_SLOTS, lag
from demand.data import REGIONS, SLOTS, WEATHER_VARS, holiday_share, load, path

FIRST_DAY = pd.Timestamp('2016-01-01')     # leaves a year of history for the slowest lags
LEVEL_DAYS = 28


def _day(values: pd.Series, index) -> np.ndarray:
    """Broadcast a per-day value over the 96 slots."""
    return np.repeat(values.reindex(index).to_numpy()[:, None], SLOTS, axis=1)


def level(actual: pd.DataFrame) -> pd.Series:
    """Mean load over the LEVEL_DAYS days ending D-2: the scale of every load feature and the target."""
    daily = actual.mean(axis=1)
    return daily.rolling(LEVEL_DAYS, min_periods=LEVEL_DAYS // 2).mean().shift(2, freq='D')


def region_features(region: str, hol: pd.DataFrame) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    a = load('actual', region)
    idx = pd.date_range(max(FIRST_DAY, a.index[0]), a.index[-1] + pd.Timedelta('1D'))
    a = a.reindex(idx)
    lev = level(a).reindex(idx)
    scaled = a.div(lev, axis=0)                           # each day's load over its own (lagged) level
    morning = a.iloc[:, :MORNING_SLOTS]
    f = {
        'target': scaled.to_numpy(),
        'level': _day(lev / lev.shift(364, freq='D').reindex(idx), idx),   # year-on-year level change
        'd2': lag(a, 2).div(lev, axis=0).to_numpy(),
        'd7': lag(a, 7).div(lev, axis=0).to_numpy(),
        'd14': lag(a, 14).div(lev, axis=0).to_numpy(),
        'd21': lag(a, 21).div(lev, axis=0).to_numpy(),
        'd364': lag(a, 364).div(lev, axis=0).to_numpy(),
        'mean_4w_slot': (sum(lag(a, k) for k in (7, 14, 21, 28)) / 4).div(lev, axis=0).to_numpy(),
        # How yesterday morning compares with the same morning a week earlier: a fresh trend signal.
        'd1_morning_vs_d8': _day(lag(morning, 1).mean(axis=1) / lag(morning, 8).mean(axis=1), idx),
        'd1_late_morning_vs_d8': _day(lag(morning.iloc[:, -8:], 1).mean(axis=1)
                                      / lag(morning.iloc[:, -8:], 8).mean(axis=1), idx),
        'd2_vs_d9': _day(lag(a, 2).mean(axis=1) / lag(a, 9).mean(axis=1), idx),
    }
    for v in WEATHER_VARS:
        if not path(f'{v}_{region}').exists():
            continue
        w = load(v, region).reindex(idx)
        f[v] = w.to_numpy()
        f[f'{v}_vs_last_week'] = (w - lag(w, 7)).to_numpy()
        f[f'{v}_day'] = _day(w.mean(axis=1), idx)
    h = hol[region]
    for name, d in [('holiday', 0), ('holiday_prev', 1), ('holiday_next', -1), ('holiday_last_week', 7)]:
        f[name] = _day(h.shift(d, freq='D'), idx)
    dow = idx.day_of_week.to_numpy()
    doy = idx.day_of_year.to_numpy()
    f['dow'] = np.repeat(dow[:, None], SLOTS, axis=1)
    f['doy_sin'] = np.repeat(np.sin(2 * np.pi * doy / 365.25)[:, None], SLOTS, axis=1)
    f['doy_cos'] = np.repeat(np.cos(2 * np.pi * doy / 365.25)[:, None], SLOTS, axis=1)
    xmas = ((idx.month == 12) & (idx.day >= 22)) | ((idx.month == 1) & (idx.day <= 6))
    f['christmas'] = np.repeat(np.asarray(xmas)[:, None], SLOTS, axis=1).astype(float)
    f['slot'] = np.tile(np.arange(SLOTS), (len(idx), 1))
    f['region'] = np.full((len(idx), SLOTS), REGIONS.index(region))
    f['scale'] = _day(lev, idx)
    return idx, f


def build() -> pd.DataFrame:
    """All regions stacked: columns day, region, slot, scale, features..., target."""
    days = pd.date_range(FIRST_DAY - pd.Timedelta('10D'), pd.Timestamp.now().normalize() + pd.Timedelta('10D'))
    hol = holiday_share(days)
    frames = []
    for region in REGIONS:
        idx, f = region_features(region, hol)
        cols = {k: v.ravel().astype(np.float32) for k, v in f.items()}
        frames.append(pd.DataFrame({'day': np.repeat(idx.to_numpy(), SLOTS), **cols}))
    df = pd.concat(frames, ignore_index=True)
    df['region'] = df['region'].astype(int)
    df['slot'] = df['slot'].astype(int)
    return df.replace([np.inf, -np.inf], np.nan)


FEATURES_EXCLUDED = {'day', 'target', 'scale'}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURES_EXCLUDED]
