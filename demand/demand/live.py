"""Live forecasting: a weekly retrain and a daily noon forecast with a running scoreboard.

    python -m demand.live train      # weekly, on the GPU: fit LightGBM and the network on all data
    python -m demand.live forecast   # daily at 12:00 Berlin: forecast tomorrow, update the scoreboard

The daily run refreshes SMARD and the weather, forecasts tomorrow's 96
quarter-hours for every region with both saved models, and appends to
live-state/forecasts.csv, together with SMARD's forecast for the same day
*as published at that moment*. Then it scores every logged day whose
actuals are complete.

With PUBLISH set, it pushes to the GitHub repo (deploy key in the CronJob):
  demand/live/forecasts/<day>.csv   that day's first forecast, never rewritten,
                                    so its commit time proves it predates the outcome
  demand/live/scoreboard.md, .json  the running score; the JSON feeds the Pages site
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from demand.data import REGIONS, SLOTS, TZ, download, load
from demand.features import build, feature_columns
from demand.quantiles import BAND_NAMES as BANDS, offsets

ROOT = Path(__file__).parent.parent
MODELS = ROOT / 'models'
LIVE = ROOT / 'live-state'                 # local working state (gitignored)
LOG = LIVE / 'forecasts.csv'
REPO = 'git@github.com:john-zither/Smard-Forecasts.git'
PUBLISHED = 'demand/live'
PROFILE_DAYS = 14                          # days of quarter-hour profiles on the website
ON_TIME = pd.Timedelta('12h30min')         # issued by 12:30 the day before counts as on time
SEEDS = 3
VAL_DAYS = 42


def train() -> None:
    import torch
    from demand.lgbm import PARAMS
    from demand.nn import fit, fit_scaler, apply_scaler, to_tensors

    # A CPU fallback is not a slower success: on 2026-11-08 the GPU wasn't visible, the
    # network trained on the CPU for two hours, and the machine ran out of memory with a
    # game open. Fail fast instead; the saved models keep serving the daily forecast.
    if not torch.cuda.is_available() and not os.environ.get('ALLOW_CPU'):
        raise SystemExit("no CUDA device; not retraining (set ALLOW_CPU=1 to train on the CPU)")
    download(incremental=True)
    df = build()
    MODELS.mkdir(exist_ok=True)

    cols = feature_columns(df)
    lab = df[df.target.notna()]
    last = lab.day.max()
    fit_rows, val_rows = lab[lab.day <= last - pd.Timedelta(f'{VAL_DAYS}D')], lab[lab.day > last - pd.Timedelta(f'{VAL_DAYS}D')]
    ds = lgb.Dataset(fit_rows[cols], fit_rows.target, categorical_feature=['region'])
    booster = lgb.train(PARAMS, ds, 2000, valid_sets=[lgb.Dataset(val_rows[cols], val_rows.target, reference=ds)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    booster.save_model(str(MODELS / 'lgbm.txt'), num_iteration=booster.best_iteration)
    print(f"lgbm: {booster.best_iteration} trees on data up to {last.date()}", flush=True)
    del ds, booster, fit_rows, val_rows, lab               # free the tabular copies first

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    days, X, y, _, nn_cols = to_tensors(df)
    del df
    labeled = np.flatnonzero(np.isfinite(y).all(axis=(1, 2)))
    train_d, val_d = labeled[:-VAL_DAYS], labeled[-VAL_DAYS:]
    sc = fit_scaler(X, train_d)
    Z = apply_scaler(X, sc)
    del X
    for seed in range(SEEDS):
        model = fit(Z, y, train_d, val_d, seed, device)
        torch.save(model.state_dict(), MODELS / f'nn_seed{seed}.pt')
        print(f"nn seed {seed} trained on {device}", flush=True)
    np.savez(MODELS / 'nn_scaler.npz', **sc)
    (MODELS / 'meta.json').write_text(json.dumps({
        'trained_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'data_until': str(last.date()), 'lgbm_features': cols, 'nn_features': nn_cols,
        'nn_inputs': int(Z.shape[-1])}, indent=2))


def predict(df: pd.DataFrame, day: pd.Timestamp) -> dict[str, np.ndarray]:
    """{'lgbm' | 'nn': [regions x 96] MWh} for one delivery day."""
    import torch
    from demand.nn import DayNet, apply_scaler, to_tensors

    meta = json.loads((MODELS / 'meta.json').read_text())
    rows = df[df.day == day].sort_values(['region', 'slot'])
    assert len(rows) == len(REGIONS) * SLOTS, f"features for {day.date()} incomplete"
    booster = lgb.Booster(model_file=str(MODELS / 'lgbm.txt'))
    scale = rows.scale.to_numpy().reshape(len(REGIONS), SLOTS)
    out = {'lgbm': booster.predict(rows[meta['lgbm_features']]).reshape(len(REGIONS), SLOTS) * scale}

    days, X, _, nn_scale, cols = to_tensors(df[df.day == day])
    assert cols == meta['nn_features']
    sc = dict(np.load(MODELS / 'nn_scaler.npz'))
    Z = torch.tensor(apply_scaler(X, sc))
    preds = []
    for seed in range(SEEDS):
        model = DayNet(meta['nn_inputs'])
        model.load_state_dict(torch.load(MODELS / f'nn_seed{seed}.pt', map_location='cpu'))
        model.eval()
        with torch.no_grad():
            preds.append(model(Z).numpy()[0])
    out['nn'] = np.mean(preds, axis=0) * nn_scale[0]
    out['ensemble'] = (out['lgbm'] + out['nn']) / 2
    return out


def residuals() -> dict[str, pd.DataFrame]:
    """{region: [day x 96] actual - ensemble}: live days from the log's first forecasts, earlier days
    from the walk-forward backtest (results/), so the bands work from the first live day."""
    from demand.ensemble import load_predictions
    try:
        backtest = load_predictions()['ensemble']
    except FileNotFoundError:
        backtest = {}
    first = scored_log() if LOG.exists() else pd.DataFrame(columns=['forecast', 'complete'])
    live = first[(first.forecast == 'ensemble') & first.complete.astype(bool)]
    out = {}
    for region in REGIONS:
        actual = load('actual', region)
        g = live[live.region == region]
        r = (g.actual - g.mwh).groupby([pd.to_datetime(g.day), g.slot]).first().unstack()
        r.columns = r.columns.astype(int)
        if region in backtest:
            r = r.combine_first(actual.reindex(backtest[region].index) - backtest[region])
        out[region] = r.sort_index()
    return out


def add_bands(fc: dict[str, np.ndarray], day: pd.Timestamp) -> None:
    """Add the q05..q95 bands (see demand.quantiles) around the ensemble, if every region has enough history."""
    resid = residuals()
    off = [offsets(resid[region], day) for region in REGIONS]
    if any(o is None for o in off):
        print("bands: not enough residual history, skipped")
        return
    for k, name in enumerate(BANDS):
        fc[name] = fc['ensemble'] + np.stack([o[k] for o in off])


def forecast() -> None:
    now = pd.Timestamp.now(TZ)
    day = (now + pd.Timedelta('1D')).normalize().tz_localize(None)
    download(incremental=True)
    df = build()
    fc = predict(df, day)
    add_bands(fc, day)
    rows = []
    for r, region in enumerate(REGIONS):
        smard = load('smard', region).reindex([day]).to_numpy()[0]
        for name, values in {**{k: v[r] for k, v in fc.items()}, 'smard': smard}.items():
            rows += [(now.isoformat(timespec='seconds'), day.date().isoformat(), region, slot, name, v)
                     for slot, v in enumerate(values)]
    LIVE.mkdir(exist_ok=True)
    new = pd.DataFrame(rows, columns=['issued_at', 'day', 'region', 'slot', 'forecast', 'mwh'])
    new.to_csv(LOG, mode='a', header=not LOG.exists(), index=False, float_format='%.1f')
    print(f"Forecast for {day.date()} issued {now:%Y-%m-%d %H:%M %Z}: "
          + ", ".join(f"{reg} {fc["ensemble"][r].sum() / 1e3:.0f} GWh" for r, reg in enumerate(REGIONS)))
    scoreboard()
    if os.environ.get('PUBLISH'):
        publish()


def scored_log() -> pd.DataFrame:
    """Each day's first logged forecast, with actual load where published."""
    log = pd.read_csv(LOG)
    first = log.groupby(['day', 'region', 'slot', 'forecast']).first().reset_index()
    actual = {region: load('actual', region) for region in REGIONS}
    first['actual'] = [actual[r].at[pd.Timestamp(d), s] if pd.Timestamp(d) in actual[r].index else np.nan
                       for d, r, s in zip(first.day, first.region, first.slot)]
    first['complete'] = first.groupby(['day', 'region']).actual.transform(lambda a: a.notna().all())
    return first


def _coverage(g: pd.DataFrame) -> dict:
    """Share of actuals inside the 50% and 90% bands, over the days that had bands."""
    w = g[g.forecast.isin(BANDS)].pivot_table(index=['day', 'slot'], columns='forecast', values='mwh')
    if set(w.columns) != set(BANDS):
        return {}
    a = g[g.forecast == 'ensemble'].set_index(['day', 'slot']).actual.reindex(w.index)
    inside = lambda lo, hi: float(((a >= w[lo]) & (a <= w[hi])).mean())
    return {'band_days': int(w.index.get_level_values('day').nunique()),
            'cover50': inside('q25', 'q75'), 'cover90': inside('q05', 'q95')}


def _nmae(g: pd.DataFrame, name: str) -> float:
    g = g[g.forecast == name]
    return float((g.mwh - g.actual).abs().mean() / g.actual.mean())


def record(now: pd.Timestamp) -> list[dict]:
    """Every day a forecast was due since the first one: when it was issued, and whether
    on time (by ON_TIME the day before), late, or missed. Missed days stay on the record."""
    log = pd.read_csv(LOG, usecols=['issued_at', 'day'])
    issued = log.groupby('day').issued_at.min()
    due_until = now.normalize().tz_localize(None) + pd.Timedelta('1D' if now.hour >= 12 else '0D')
    out = []
    for day in pd.date_range(pd.Timestamp(issued.index.min()), due_until):
        key = day.date().isoformat()
        if key not in issued:
            out.append({'day': key, 'issued_at': None, 'status': 'missed'})
            continue
        at = pd.Timestamp(issued[key]).tz_convert(TZ)
        deadline = (day - pd.Timedelta('1D')).tz_localize(TZ) + ON_TIME
        out.append({'day': key, 'issued_at': at.isoformat(timespec='minutes'),
                    'status': 'on_time' if at <= deadline else 'late'})
    return out


def scoreboard() -> None:
    """Score each logged day (its first forecast only) once its actuals are complete."""
    first = scored_log()
    done = first[first.complete]
    lines = ["# Live scoreboard", "",
             "Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to",
             "`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.", ""]
    now = pd.Timestamp.now(TZ)
    board = {'updated_at': now.isoformat(timespec='seconds'), 'regions': REGIONS,
             'live': {}, 'daily': [], 'profiles': [], 'record': record(now), 'model': None}
    if (MODELS / 'meta.json').exists():
        meta = json.loads((MODELS / 'meta.json').read_text())
        board['model'] = {k: meta[k] for k in ('trained_at', 'data_until')}
    if done.empty:
        lines.append("No complete day yet.")
    else:
        lines += [f"{done.day.nunique()} day(s) scored, {done.day.min()} → {done.day.max()}.", "",
                  "| region | this model | SMARD | error vs SMARD |", "|---|---|---|---|"]
        for region in REGIONS:
            g = done[done.region == region]
            e, s = _nmae(g, 'ensemble'), _nmae(g, 'smard')
            board['live'][region] = {'days': int(g.day.nunique()), 'model': e, 'smard': s} | _coverage(g)
            lines.append(f"| {region} | {e:.2%} | {s:.2%} | {e / s - 1:+.0%} |")
        cov = {r: v for r, v in board['live'].items() if 'cover90' in v}
        if cov:
            lines += ["", "Uncertainty bands, share of quarter-hours inside (target 50% / 90%):", "",
                      "| region | days | 50% band | 90% band |", "|---|---|---|---|"]
            lines += [f"| {r} | {v['band_days']} | {v['cover50']:.0%} | {v['cover90']:.0%} |" for r, v in cov.items()]
        for (day, region), g in done.groupby(['day', 'region']):
            board['daily'].append({'day': day, 'region': region, 'model': _nmae(g, 'ensemble'), 'smard': _nmae(g, 'smard')})
    for (day, region), g in first[first.day.isin(sorted(first.day.unique())[-PROFILE_DAYS:])].groupby(['day', 'region']):
        prof = {name: [round(float(v), 1) for v in g[g.forecast == name].sort_values('slot').mwh]
                for name in ('ensemble', 'smard', *BANDS) if (g.forecast == name).any()}
        act = g[g.forecast == 'ensemble'].sort_values('slot').actual
        prof['actual'] = [None if pd.isna(v) else round(float(v), 1) for v in act]
        board['profiles'].append({'day': day, 'region': region, **prof})
    (LIVE / 'scoreboard.md').write_text("\n".join(lines) + "\n")
    (LIVE / 'scoreboard.json').write_text(json.dumps(board, separators=(',', ':')))
    print("\n".join(lines[-7:]))


def publish() -> None:
    """Commit the new day files and the scoreboard to GitHub (sparse, shallow clone in /tmp)."""
    log = pd.read_csv(LOG)
    with tempfile.TemporaryDirectory() as tmp:
        git = lambda *a: subprocess.run(['git', '-C', tmp, *a], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse', REPO, tmp],
                       check=True, capture_output=True, text=True)
        git('sparse-checkout', 'set', PUBLISHED)
        out = Path(tmp) / PUBLISHED
        (out / 'forecasts').mkdir(parents=True, exist_ok=True)
        new_days = []
        for day, g in log.groupby('day'):
            f = out / 'forecasts' / f'{day}.csv'
            if not f.exists():                        # a day's file is never rewritten
                first_issue = g.issued_at.min()
                g[g.issued_at == first_issue].to_csv(f, index=False, float_format='%.1f')
                new_days.append(day)
        for name in ('scoreboard.md', 'scoreboard.json'):
            shutil.copy(LIVE / name, out / name)
        git('add', PUBLISHED)
        if not git('status', '--porcelain').stdout.strip():
            print("publish: nothing new")
            return
        git('-c', 'user.name=John Theurer', '-c', 'user.email=john@energy-data-solutions.com',
            'commit', '-m', f"Live forecast for {', '.join(new_days)}" if new_days else "Update live scoreboard")
        git('push', 'origin', 'HEAD:main')
        print(f"publish: pushed {', '.join(new_days) or 'scoreboard'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=['train', 'forecast', 'scoreboard', 'publish'])
    {'train': train, 'forecast': forecast, 'scoreboard': scoreboard, 'publish': publish}[parser.parse_args().command]()


if __name__ == '__main__':
    main()
