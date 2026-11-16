"""Why are SMARD's regional demand forecasts poor? Reproduces the tables in results/why-smard-is-off.md.

    uv run python -m demand.diagnose

Uses only the cached SMARD actuals/forecasts and Open-Meteo weather in data/.
"""

import numpy as np
import pandas as pd

from .data import REGIONS, load

DAYTIME = slice(36, 64)                 # 09:00-16:00 wall clock
MIDDAY = slice(40, 56)                  # 10:00-14:00


def pair(region: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Actual and SMARD forecast in MW, days where both are complete."""
    a, f = load('actual', region), load('smard', region)
    ix = a.index.intersection(f.index)
    a, f = a.loc[ix] * 4, f.loc[ix] * 4
    ok = a.notna().all(axis=1) & f.notna().all(axis=1)
    return a[ok], f[ok]


def yearly_bias() -> None:
    print('Mean error (forecast - actual) as % of load, by year')
    rows = {}
    for r in REGIONS:
        a, f = pair(r)
        rows[r] = ((f - a).mean(axis=1) / a.mean(axis=1) * 100).groupby(a.index.year).mean()
    print(pd.DataFrame(rows).T.round(1).to_string(), '\n')


def weather_response(since: str = '2024-01-01') -> None:
    print(f'How midday load responds to sun and cold, actual vs forecast (since {since})')
    print('regression of 10-14h load on temperature, midday radiation, weekday and year dummies')
    for r in REGIONS:
        a, f = pair(r)
        t = load('temperature_2m', r).reindex(a.index).mean(axis=1)
        sun = load('shortwave_radiation', r).reindex(a.index).iloc[:, MIDDAY].mean(axis=1)
        ok = (a.index >= since) & t.notna().to_numpy() & sun.notna().to_numpy()
        ix = a.index[ok]
        X = np.column_stack([np.minimum(t[ok], 15), np.maximum(t[ok], 15), sun[ok]]
                            + [(ix.day_of_week == k) for k in range(7)]
                            + [(ix.year == y) for y in sorted(set(ix.year))[1:]]).astype(float)
        (ca, _, sa), (cf, _, sf) = (np.linalg.lstsq(X, y.iloc[:, MIDDAY].mean(axis=1)[ok], rcond=None)[0][:3]
                                    for y in (a, f))
        print(f'  {r:10s} sun: actual {sa:5.1f}, forecast {sf:5.1f} MW per W/m2 ({sf / sa:4.0%} captured)'
              f' | cold: actual {ca:5.0f}, forecast {cf:5.0f} MW per degC')
    print()


def sunday_ratio(region: str = '50Hertz') -> None:
    print(f'{region}: daytime (09-16h) load on Sunday / the Saturday before, quarterly mean')
    a, f = pair(region)
    out = {}
    for name, x in (('actual', a), ('forecast', f)):
        d = x.iloc[:, DAYTIME].mean(axis=1)
        sun = d[d.index.day_of_week == 6]
        sat = d.reindex(sun.index - pd.Timedelta('1D')).to_numpy()
        out[name] = (sun / sat).resample('QS').mean()
    print(pd.DataFrame(out)['2023':].round(3).to_string(), '\n')


def transnetbw_step() -> None:
    print('TransnetBW: weekly mean load (MW) around the October 2020 step')
    a, f = pair('TransnetBW')
    w = pd.DataFrame({'actual': a.mean(axis=1), 'forecast': f.mean(axis=1)})
    w['gap'] = w.forecast - w.actual
    print(w['2020-09-01':'2020-11-15'].resample('W-MON').mean().round(0).to_string(), '\n')
    print('TransnetBW: monthly gap (MW) in 2026, fixed from August')
    print(w.gap['2026-01':].resample('MS').mean().round(0).to_string(), '\n')


def correctable(test_from: str = '2025-01-01') -> None:
    print(f'nMAE % from {test_from}: SMARD as published, and minus its own trailing 28-day per-slot bias')
    for r in REGIONS:
        a, f = pair(r)
        bias = (f - a).rolling('28D').mean().shift(2)     # actuals are published with a delay
        ok = (a.index >= test_from) & bias.notna().all(axis=1).to_numpy()
        n = lambda g: (g[ok] - a[ok]).abs().to_numpy().mean() / a[ok].to_numpy().mean() * 100
        print(f'  {r:10s} {n(f):4.1f} -> {n(f - bias):4.1f}')


if __name__ == '__main__':
    yearly_bias()
    weather_response()
    sunday_ratio()
    transnetbw_step()
    correctable()
