"""Probabilistic demand forecast: 50% and 90% ranges around the point forecast.

A point forecast says what load to expect. Grid operators and traders also need
to know how wrong it might be, so the live forecast publishes the 5/25/75/95%
quantiles per quarter-hour.

Two methods, scored in the same monthly walk-forward as demand.lgbm:

  empirical (published live)
      ensemble point forecast + the empirical quantiles of its own residuals
      over the last 90 complete days (up to D-2), pooled per 4-hour block.
  LightGBM quantiles (tried, not adopted)
      one LightGBM per quantile (pinball loss), early-stopped on the 42 days
      before the month, then conformally shifted on those days so coverage
      hits the target. Scored both as-is and re-centred on the ensemble.

The simple method won in every region: lower pinball loss and a narrower 90%
range at similar coverage (results/quantiles.md). The quantile trees stop early
on the tails (100-300 trees at q=0.95) and need large conformal shifts, so their
learnt weather-dependent width doesn't make up for the noise.

    uv run python -m demand.quantiles           # ~1 h on 6 CPU threads; resumes per month
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from demand.data import REGIONS, SLOTS, load
from demand.ensemble import load_predictions
from demand.evaluate import RESULTS, TEST_END, TEST_START
from demand.features import build, feature_columns
from demand.lgbm import PARAMS, VAL_DAYS

QUANTILES = (0.05, 0.25, 0.75, 0.95)
BAND_NAMES = tuple(f'q{round(q * 100):02d}' for q in QUANTILES)      # forecast names in the live log
BAND_DAYS = 90                          # residual window; 28/42/63/90/120/180 tried, chosen on Dec-May, held on Jun-Oct
BLOCKS = 6                              # 4-hour blocks; hourly (24) was slightly worse
MIN_BAND_DAYS = 20
INTERVALS = {'50%': (0.25, 0.75), '90%': (0.05, 0.95)}
CACHE = RESULTS / 'quantiles'
EVAL_START = TEST_START + pd.offsets.MonthBegin(1)    # the empirical baseline needs a month of ensemble residuals


def params(q: float) -> dict:
    return PARAMS | {'objective': 'quantile', 'alpha': q, 'metric': 'quantile'}


def fit_month(df: pd.DataFrame, cols: list[str], month: pd.Timestamp) -> pd.DataFrame:
    """Conformal-shifted quantiles of scaled load (and the l1 median) for one test month."""
    cutoff = month - pd.Timedelta('2D')
    train = df[(df.day <= cutoff) & df.target.notna()]
    edge = cutoff - pd.Timedelta(f'{VAL_DAYS}D')
    fit, val = train[train.day <= edge], train[train.day > edge]
    test = df[(df.day >= month) & (df.day < month + pd.offsets.MonthBegin(1))]
    ds = lgb.Dataset(fit[cols], fit.target, categorical_feature=['region'], free_raw_data=False)
    out = pd.DataFrame(index=test.index)
    for q in (0.5, *QUANTILES):
        p = PARAMS if q == 0.5 else params(q)
        dv = lgb.Dataset(val[cols], val.target, reference=ds)
        m = lgb.train(p, ds, 2000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
        pv = m.predict(val[cols], num_iteration=m.best_iteration)
        shift = 0.0 if q == 0.5 else np.quantile(val.target - pv, q)
        out[q] = m.predict(test[cols], num_iteration=m.best_iteration) + shift
        print(f"  {month:%Y-%m} q={q:.2f}: {m.best_iteration} trees, conformal shift {shift:+.4f}", flush=True)
    return out


def walk_forward() -> pd.DataFrame:
    """[test rows x quantile] in MWh per quarter-hour, with day/region/slot; cached per month."""
    df = build()
    cols = feature_columns(df)
    CACHE.mkdir(parents=True, exist_ok=True)
    parts = []
    for month in pd.date_range(TEST_START, TEST_END, freq='MS'):
        f = CACHE / f'{month:%Y-%m}.parquet'
        if not f.exists():
            fit_month(df, cols, month).to_parquet(f)
        parts.append(pd.read_parquet(f))
    q = pd.concat(parts)
    q.columns = q.columns.astype(float)
    meta = df.loc[q.index, ['day', 'region', 'slot', 'scale']]
    return pd.concat([meta, q.mul(meta.scale, axis=0)], axis=1)


def matrices(q: pd.DataFrame) -> dict[str, dict[float, pd.DataFrame]]:
    """{region: {quantile: [day x 96]}}; the l1 median is under 0.5."""
    out = {}
    for r, region in enumerate(REGIONS):
        part = q[q.region == r].sort_values(['day', 'slot'])
        days = pd.DatetimeIndex(part.day.iloc[::SLOTS])
        out[region] = {k: pd.DataFrame(part[k].to_numpy().reshape(-1, SLOTS), index=days) for k in (0.5, *QUANTILES)}
    return out


def offsets(resid: pd.DataFrame, day: pd.Timestamp) -> np.ndarray | None:
    """[quantile x 96] offsets to add to the point forecast for `day`: the empirical quantiles of the
    residuals (actual - forecast) of the BAND_DAYS complete days up to D-2, pooled per 4-hour block.
    None if fewer than MIN_BAND_DAYS such days exist."""
    past = resid[(resid.index <= day - pd.Timedelta('2D'))
                 & (resid.index > day - pd.Timedelta(f'{BAND_DAYS + 2}D'))].dropna()
    if len(past) < MIN_BAND_DAYS:
        return None
    width = SLOTS // BLOCKS
    out = np.empty((len(QUANTILES), SLOTS))
    for b in range(BLOCKS):
        cols = slice(b * width, (b + 1) * width)
        out[:, cols] = np.quantile(past.iloc[:, cols].to_numpy().ravel(), QUANTILES)[:, None]
    return out


def empirical(ens: pd.DataFrame, actual: pd.DataFrame) -> dict[float, pd.DataFrame]:
    """The empirical band around the ensemble for every day, as published live."""
    resid = actual.reindex(ens.index) - ens
    out = {q: pd.DataFrame(np.nan, index=ens.index, columns=ens.columns) for q in QUANTILES}
    for d in ens.index:
        off = offsets(resid, d)
        if off is not None:
            for k, q in enumerate(QUANTILES):
                out[q].loc[d] = ens.loc[d].to_numpy() + off[k]
    return out


def scores(qs: dict[float, pd.DataFrame], point: pd.DataFrame, actual: pd.DataFrame) -> dict:
    days = pd.date_range(EVAL_START, TEST_END)
    a = actual.reindex(days).to_numpy()
    Q = {q: m.reindex(days).to_numpy() for q, m in qs.items()}
    ok = np.isfinite(a) & np.all([np.isfinite(v) for v in Q.values()], axis=0)
    mean = a[ok].mean()
    out = {}
    for name, (lo, hi) in INTERVALS.items():
        out[f'cover {name}'] = ((a >= Q[lo]) & (a <= Q[hi]))[ok].mean()
        out[f'width {name}'] = (Q[hi] - Q[lo])[ok].mean() / mean
    pin = [np.maximum(q * (a - Q[q]), (q - 1) * (a - Q[q]))[ok].mean() for q in QUANTILES]
    out['pinball'] = np.mean(pin) / mean
    out['nMAE'] = np.abs(point.reindex(days).to_numpy() - a)[ok].mean() / mean
    return out


def main() -> None:
    q = walk_forward()
    lg = matrices(q)
    ens = load_predictions()['ensemble']
    rows = []
    for region in REGIONS:
        actual = load('actual', region)
        e = ens[region]
        med = lg[region][0.5].reindex(e.index)
        methods = {
            'empirical (ensemble + 90-day residuals)': (empirical(e, actual), e),
            'LightGBM quantiles': ({k: lg[region][k] for k in QUANTILES}, lg[region][0.5]),
            'LightGBM quantiles on ensemble': ({k: e + lg[region][k].reindex(e.index) - med for k in QUANTILES}, e),
        }
        for name, (qs, point) in methods.items():
            rows.append({'region': region, 'method': name, **scores(qs, point, actual)})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / 'quantiles.csv', index=False)
    cols = list(res.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|---' * len(cols) + '|']
    for row in res.itertuples(index=False):
        lines.append('| ' + ' | '.join([row[0], row[1]] + [f'{v:.2%}' for v in row[2:]]) + ' |')
    text = (f"Quantile forecasts, {EVAL_START.date()} → {TEST_END.date()}, issue time 12:00 D-1, monthly walk-forward.\n"
            f"cover = share of quarter-hours inside the interval (target 50% / 90%); width and pinball loss are "
            f"relative to mean load (lower is better, pinball averaged over q = {', '.join(map(str, QUANTILES))}).\n\n"
            + '\n'.join(lines))
    print(text)
    (RESULTS / 'quantiles.md').write_text(text + '\n')


if __name__ == '__main__':
    main()
