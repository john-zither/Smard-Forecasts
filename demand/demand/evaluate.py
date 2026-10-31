"""Score day-ahead demand forecasts per region against actual load.

Metrics, over every quarter-hour of the test window:
  nMAE     MAE / mean actual load (the headline number; scale-free across regions)
  MAPE     mean absolute percentage error, as SMARD-style comparisons usually quote
  vs SMARD 1 - MAE / SMARD's MAE on the same quarter-hours (> 0 beats SMARD)

    uv run python -m demand.evaluate          # baselines only
"""

from pathlib import Path

import numpy as np
import pandas as pd

from demand.baselines import baselines
from demand.data import REGIONS, load

RESULTS = Path(__file__).parent.parent / 'results'
TEST_START, TEST_END = pd.Timestamp('2025-11-01'), pd.Timestamp('2026-10-30')   # inclusive days


def score(forecast: pd.DataFrame, actual: pd.DataFrame, smard: pd.DataFrame) -> dict:
    days = pd.date_range(TEST_START, TEST_END)
    f, a, s = (m.reindex(days).to_numpy() for m in (forecast, actual, smard))
    ok = np.isfinite(f) & np.isfinite(a) & np.isfinite(s)
    err, serr = np.abs(f - a)[ok], np.abs(s - a)[ok]
    return {'quarter_hours': int(ok.sum()), 'nMAE': err.mean() / a[ok].mean(),
            'MAPE': (err / a[ok]).mean(), 'vs_smard': 1 - err.mean() / serr.mean()}


def run(models: dict[str, dict[str, pd.DataFrame]] | None = None) -> pd.DataFrame:
    """Score baselines plus models ({name: {region: [day x 96]}}) for every region."""
    rows = []
    for region in REGIONS:
        actual, smard = load('actual', region), load('smard', region)
        fc = baselines(actual, smard) | {name: m[region] for name, m in (models or {}).items() if region in m}
        for name, f in fc.items():
            rows.append({'region': region, 'forecast': name, **score(f, actual, smard)})
    return pd.DataFrame(rows)


def monthly(forecast: dict[str, pd.DataFrame], name: str) -> str:
    """Markdown: per month and region, the forecast's nMAE next to SMARD's."""
    days = pd.date_range(TEST_START, TEST_END)
    lines = ['| month | ' + ' | '.join(f'{r} {name} / SMARD' for r in REGIONS) + ' |', '|---' * (len(REGIONS) + 1) + '|']
    cells = {}
    for region in REGIONS:
        a, s = load('actual', region).reindex(days), load('smard', region).reindex(days)
        f = forecast[region].reindex(days)
        for month, idx in a.groupby(a.index.to_period('M')).groups.items():
            ok = np.isfinite(f.loc[idx]) & np.isfinite(a.loc[idx]) & np.isfinite(s.loc[idx])
            mean = a.loc[idx][ok].stack().mean()
            cells[(str(month), region)] = (np.abs(f.loc[idx] - a.loc[idx])[ok].stack().mean() / mean,
                                           np.abs(s.loc[idx] - a.loc[idx])[ok].stack().mean() / mean)
    for month in sorted({m for m, _ in cells}):
        lines.append(f'| {month} | ' + ' | '.join(f'{cells[(month, r)][0]:.1%} / {cells[(month, r)][1]:.1%}'
                                                  for r in REGIONS) + ' |')
    return '\n'.join(lines)


def table(df: pd.DataFrame) -> str:
    """Markdown table: one row per region, nMAE per forecast, best non-SMARD in bold."""
    piv = df.pivot(index='region', columns='forecast', values='nMAE').reindex(REGIONS)
    cols = list(dict.fromkeys(df['forecast']))
    lines = ['| region | ' + ' | '.join(cols) + ' |', '|---' * (len(cols) + 1) + '|']
    for region, row in piv.iterrows():
        best = row[cols].idxmin()
        cells = [f"**{row[c]:.2%}**" if c == best else f"{row[c]:.2%}" for c in cols]
        lines.append(f'| {region} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main() -> None:
    df = run()
    text = (f"nMAE (MAE / mean load), {TEST_START.date()} → {TEST_END.date()}, issue time 12:00 D-1\n\n"
            + table(df))
    print(text)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'baselines.md').write_text(text + '\n')


if __name__ == '__main__':
    main()
