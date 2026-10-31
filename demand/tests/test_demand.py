from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import demand.features as features
from demand.baselines import MORNING_SLOTS, baselines
from demand.data import REGIONS, SLOTS, holiday_share


def _fake_load(actual):
    return lambda name, region: actual.copy()


def _features(monkeypatch, actual, day):
    monkeypatch.setattr(features, 'load', _fake_load(actual))
    monkeypatch.setattr(features, 'path', lambda name: Path('/nonexistent'))
    monkeypatch.setattr(features, 'FIRST_DAY', actual.index[0])
    hol = holiday_share(pd.date_range(actual.index[0] - pd.Timedelta('10D'), actual.index[-1] + pd.Timedelta('10D')))
    idx, f = features.region_features('TenneT', hol)
    i = idx.get_loc(day)
    return {k: v[i] for k, v in f.items() if k != 'target'}


def test_features_use_only_what_is_published_by_noon(monkeypatch):
    days = pd.date_range('2024-01-01', '2025-03-31')
    rng = np.random.default_rng(0)
    actual = pd.DataFrame(1000 + rng.normal(0, 50, (len(days), SLOTS)), index=days)
    day = pd.Timestamp('2025-03-20')
    before = _features(monkeypatch, actual, day)
    changed = actual.copy()
    changed.loc[day:] += 500                                      # the day being forecast, and later
    changed.loc[day - pd.Timedelta('1D'), MORNING_SLOTS:] += 500  # yesterday after 10:15
    after = _features(monkeypatch, changed, day)
    for k in before:
        np.testing.assert_allclose(before[k], after[k], err_msg=k, equal_nan=True)


def test_features_do_see_yesterday_morning(monkeypatch):
    days = pd.date_range('2024-01-01', '2025-03-31')
    actual = pd.DataFrame(1000.0, index=days, columns=range(SLOTS))
    day = pd.Timestamp('2025-03-20')
    changed = actual.copy()
    changed.loc[day - pd.Timedelta('1D'), :MORNING_SLOTS - 1] *= 1.1
    assert _features(monkeypatch, changed, day)['d1_morning_vs_d8'][0] == pytest.approx(1.1)


def test_baselines_lag_correctly():
    days = pd.date_range('2025-01-01', periods=60)
    actual = pd.DataFrame(np.repeat(np.arange(60.0)[:, None], SLOTS, axis=1), index=days)
    smard = actual + 10.0                                         # always 10 too high
    b = baselines(actual, smard)
    assert b['last_week'].iloc[30, 0] == 23
    assert b['smard_debiased'].iloc[45, 0] == pytest.approx(45)  # bias removed exactly


def test_holiday_share_is_regional():
    h = holiday_share(pd.DatetimeIndex(['2025-01-06', '2025-12-25']))   # Epiphany: BW, BY, ST only
    assert h.loc['2025-12-25'].eq(1).all()
    assert h.loc['2025-01-06', 'TransnetBW'] == 1 and h.loc['2025-01-06', 'Amprion'] < 0.1
