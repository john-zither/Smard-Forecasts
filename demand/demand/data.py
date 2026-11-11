"""SMARD demand actuals and forecasts, weather forecasts and holidays.

Everything is stored as [day x 96 quarter-hour] matrices on the local
(Europe/Berlin) wall clock, one per region: the forecast is issued once a
day for a Berlin calendar day, and a fixed shape keeps the modelling code
free of DST special cases. On the spring DST day the missing hour repeats
the hour before; on the autumn day the repeated hour keeps its first
occurrence.

Sources:
  SMARD filter 410  actual grid load ("Netzlast"), per TSO region and DE
  SMARD filter 411  SMARD's published day-ahead load forecast
  Open-Meteo previous-runs API, *_previous_day2: what the weather model
      forecast two days earlier. That is honest at a noon D-1 issue time (a
      noon forecast is fresher, so this slightly understates what's possible).
      Temperature from 2021, radiation and cloud cover from 2024.

    uv run python -m demand.data
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import time

import holidays
import numpy as np
import pandas as pd
import requests

SMARD = 'https://www.smard.de/app/chart_data'
METEO = 'https://previous-runs-api.open-meteo.com/v1/forecast'
TZ = 'Europe/Berlin'
SLOTS = 96
REGIONS = ['DE', '50Hertz', 'Amprion', 'TenneT', 'TransnetBW']
TSOS = REGIONS[1:]
ACTUAL, FORECAST = 410, 411
DATA_DIR = Path(__file__).parent.parent / 'data'
WEATHER_START = '2021-01-01'
REFETCH_DAYS = 14                       # the current year's weather is refetched this far back
WEATHER_VARS = ['temperature_2m', 'shortwave_radiation', 'cloud_cover']

# Population-weighted points per TSO region (largest cities, population in millions).
CITIES = {
    '50Hertz': [(52.52, 13.40, 3.7), (53.55, 10.00, 1.9), (51.34, 12.37, 0.6), (51.05, 13.74, 0.56),
                (52.13, 11.63, 0.24), (50.98, 11.03, 0.21), (54.09, 12.10, 0.21)],
    'Amprion': [(50.94, 6.96, 1.1), (51.23, 6.78, 0.62), (51.51, 7.47, 0.59), (51.46, 7.01, 0.58),
                (51.96, 7.63, 0.32), (50.36, 7.59, 0.11), (49.24, 6.99, 0.18)],
    'TenneT': [(48.14, 11.58, 1.5), (50.11, 8.68, 0.77), (52.37, 9.73, 0.54), (49.45, 11.08, 0.52),
               (53.08, 8.80, 0.57), (54.32, 10.13, 0.25), (51.31, 9.49, 0.2)],
    'TransnetBW': [(48.78, 9.18, 0.63), (49.01, 8.40, 0.31), (49.49, 8.47, 0.31), (47.99, 7.84, 0.23),
                   (48.40, 9.99, 0.13), (49.14, 9.22, 0.13)],
}
# Rough share of each TSO region's population by federal state, for holiday features.
STATES = {
    '50Hertz': {'BE': .25, 'BB': .15, 'HH': .11, 'MV': .1, 'SN': .2, 'ST': .1, 'TH': .09},
    'Amprion': {'NW': .75, 'RP': .15, 'SL': .04, 'HE': .03, 'BW': .03},
    'TenneT': {'BY': .48, 'NI': .28, 'HE': .14, 'SH': .1},
    'TransnetBW': {'BW': 1.0},
}
STATES['DE'] = {}
for _tso in TSOS:                       # states split between regions (HE, BW) add up
    for _s, _w in STATES[_tso].items():
        STATES['DE'][_s] = STATES['DE'].get(_s, 0) + _w / 4


def _get(url: str, **params) -> dict:
    for attempt in range(8):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code != 429:
                r.raise_for_status()
                return r.json()
        except (requests.Timeout, requests.ConnectionError, requests.JSONDecodeError):
            if attempt == 7:                    # slow, unreachable, or an empty/truncated 200
                raise
        time.sleep(60 * (attempt + 1))      # Open-Meteo's free tier limits calls per minute and hour
    r.raise_for_status()


def _wall_clock_days(utc: pd.Series) -> pd.DataFrame:
    """UTC 15-minute series -> [Berlin day x 96] matrix of complete days."""
    local = utc.tz_convert(TZ).tz_localize(None)
    local = local[~local.index.duplicated(keep='first')]
    grid = pd.date_range(local.index[0].normalize(), local.index[-1].normalize() + pd.Timedelta('1D'),
                         freq='15min', inclusive='left')
    s = local.reindex(grid).ffill(limit=4)
    m = s.to_numpy().reshape(-1, SLOTS)
    return pd.DataFrame(m, index=grid[::SLOTS])


def smard_series(filter_id: int, region: str, recent: int | None = None) -> pd.DataFrame:
    """Full history, or only the last `recent` weekly chunks."""
    root = f'{SMARD}/{filter_id}/{region}'
    chunks = _get(f'{root}/index_quarterhour.json')['timestamps'][-(recent or 0):]
    with ThreadPoolExecutor(8) as pool:
        parts = pool.map(lambda ts: _get(f'{root}/{filter_id}_{region}_quarterhour_{ts}.json')['series'], chunks)
        rows = [(t, v) for part in parts for t, v in part if v is not None]
    s = pd.Series([v for _, v in rows], index=pd.to_datetime([t for t, _ in rows], unit='ms', utc=True))
    s = s.sort_index()
    return _wall_clock_days(s[~s.index.duplicated()])


def weather(end: str) -> dict[str, pd.DataFrame]:
    """Population-weighted forecast weather: {(region, var): [day x 96]}."""
    points = [(r, lat, lon, w) for r, cs in CITIES.items() for lat, lon, w in cs]
    frames = []
    raw_dir = DATA_DIR / 'weather_raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    years = pd.date_range(WEATHER_START, end, freq='YS').append(pd.DatetimeIndex([pd.Timestamp(end)]))
    for lo, hi in zip(years[:-1], years[1:]):
        cache = raw_dir / f'{lo.year}.parquet'
        complete = hi.month == 1 and hi.day == 1          # past years never change; refetch the current one
        old = pd.read_parquet(cache) if cache.exists() else None
        if complete and old is not None:
            frames.append(old)
            continue
        # Open-Meteo bills a request by locations x fortnights, so the full year so far costs
        # hundreds of calls; refetch only the last REFETCH_DAYS of a cached current year.
        start = lo
        if old is not None and len(old):
            start = max(lo, old.index.max().tz_localize(None).normalize() - pd.Timedelta(f'{REFETCH_DAYS}D'))
        res = _get(METEO, latitude=','.join(f'{p[1]}' for p in points),
                   longitude=','.join(f'{p[2]}' for p in points),
                   hourly=','.join(f'{v}_previous_day2' for v in WEATHER_VARS),
                   start_date=start.date().isoformat(), end_date=(hi - pd.Timedelta('1D')).date().isoformat(),
                   timezone='UTC')
        year = []
        for (region, _, _, w), loc in zip(points, res):
            h = pd.DataFrame(loc['hourly'])
            h.index = pd.to_datetime(h.pop('time'), utc=True)
            h.columns = [c.removesuffix('_previous_day2') for c in h.columns]
            year.append(h.assign(region=region, w=w))
        new = pd.concat(year)
        if start > lo:
            new = pd.concat([old[old.index < start.tz_localize('UTC')], new])
        frames.append(new)
        frames[-1].to_parquet(cache)
        print(f"  weather {lo.year}", flush=True)
    df = pd.concat(frames)
    df[WEATHER_VARS] = df[WEATHER_VARS].astype(float)     # all-null columns arrive as object
    out = {}
    for region in REGIONS:
        part = df if region == 'DE' else df[df.region == region]
        weighted = part[WEATHER_VARS].mul(part.w, axis=0).groupby(level=0).sum(min_count=1)
        weighted = weighted.div(part.w.groupby(level=0).sum(), axis=0)
        weighted[part[WEATHER_VARS].isna().groupby(level=0).any().any(axis=1)] = np.nan
        q = weighted.resample('15min').interpolate(limit=4)
        for v in WEATHER_VARS:
            out[(region, v)] = _wall_clock_days(q[v])
    return out


def holiday_share(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Population-weighted share of each region on a public holiday, per day."""
    years = range(days.year.min(), days.year.max() + 1)
    cal = {s: holidays.Germany(subdiv=s, years=years) for s in {s for m in STATES.values() for s in m}}
    return pd.DataFrame({r: [sum(w for s, w in STATES[r].items() if d in cal[s]) for d in days.date]
                         for r in REGIONS}, index=days)


def path(name: str) -> Path:
    return DATA_DIR / f'{name}.parquet'


def download(smard: bool = True, incremental: bool = False) -> None:
    """incremental: refetch only the last three weeks of SMARD data and merge them in."""
    DATA_DIR.mkdir(exist_ok=True)
    for region in REGIONS if smard else []:
        for fid, name in [(ACTUAL, 'actual'), (FORECAST, 'smard')]:
            if incremental and path(f'{name}_{region}').exists():
                m = smard_series(fid, region, recent=3).combine_first(load(name, region))
            else:
                m = smard_series(fid, region)
            m.columns = m.columns.astype(str)
            m.to_parquet(path(f'{name}_{region}'))
            print(f"  {name:6s} {region:10s} {m.index[0].date()} → {m.index[-1].date()}")
    end = (pd.Timestamp.now(TZ).normalize() + pd.Timedelta('3D')).tz_localize(None)
    for (region, var), m in weather(end.date().isoformat()).items():
        m.columns = m.columns.astype(str)
        m.to_parquet(path(f'{var}_{region}'))
    print(f"  weather {WEATHER_START} → {end.date()} for {len(REGIONS)} regions")


def load(name: str, region: str) -> pd.DataFrame:
    m = pd.read_parquet(path(f'{name}_{region}'))
    m.columns = m.columns.astype(int)
    return m


if __name__ == '__main__':
    import sys
    download(smard='--weather-only' not in sys.argv)
