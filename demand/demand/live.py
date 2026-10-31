"""Live forecasting: a weekly retrain and a daily noon forecast with a running scoreboard.

    python -m demand.live train      # weekly, on the GPU: fit LightGBM and the network on all data
    python -m demand.live forecast   # daily at 12:00 Berlin: forecast tomorrow, update the scoreboard

The daily run refreshes SMARD and the weather, forecasts tomorrow's 96
quarter-hours for every region with both saved models, and appends to
live/forecasts.csv, together with SMARD's forecast for the same day *as
published at that moment*. Then it scores every logged day whose actuals
are complete: live/scoreboard.md. Each row records when it was issued, so
the log is a record of forecasts made before the outcome was known.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from demand.data import REGIONS, SLOTS, TZ, download, load
from demand.features import build, feature_columns

ROOT = Path(__file__).parent.parent
MODELS = ROOT / 'models'
LIVE = ROOT / 'live'
LOG = LIVE / 'forecasts.csv'
SEEDS = 3
VAL_DAYS = 42


def train() -> None:
    import torch
    from demand.lgbm import PARAMS
    from demand.nn import fit, fit_scaler, apply_scaler, to_tensors

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    days, X, y, _, nn_cols = to_tensors(df)
    labeled = np.flatnonzero(np.isfinite(y).all(axis=(1, 2)))
    train_d, val_d = labeled[:-VAL_DAYS], labeled[-VAL_DAYS:]
    sc = fit_scaler(X, train_d)
    Z = apply_scaler(X, sc)
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


def forecast() -> None:
    now = pd.Timestamp.now(TZ)
    day = (now + pd.Timedelta('1D')).normalize().tz_localize(None)
    download(incremental=True)
    df = build()
    fc = predict(df, day)
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


def scoreboard() -> None:
    """Score each logged day (its first forecast only) once its actuals are complete."""
    log = pd.read_csv(LOG)
    first = log.groupby(['day', 'region', 'slot', 'forecast']).first().reset_index()
    actual = {region: load('actual', region) for region in REGIONS}
    first['actual'] = [actual[r].at[pd.Timestamp(d), s] if pd.Timestamp(d) in actual[r].index else np.nan
                       for d, r, s in zip(first.day, first.region, first.slot)]
    first = first[first.groupby(['day', 'region']).actual.transform(lambda a: a.notna().all())]
    lines = ["# Live scoreboard", "",
             "Forecasts issued at 12:00 (Europe/Berlin) the day before, logged in `forecasts.csv` before the",
             "outcome was known. nMAE = mean absolute error / mean actual load, over all scored quarter-hours.", ""]
    if first.empty:
        lines.append("No complete day yet.")
    else:
        days = first.day.nunique()
        lines += [f"{days} day(s) scored, {first.day.min()} → {first.day.max()}.", "",
                  "| region | this model | SMARD | error vs SMARD |", "|---|---|---|---|"]
        for region in REGIONS:
            g = first[first.region == region]
            err = lambda name: (g[g.forecast == name].mwh - g[g.forecast == name].actual).abs().mean()
            mean = g[g.forecast == 'ensemble'].actual.mean()
            e, s = err('ensemble'), err('smard')
            lines.append(f"| {region} | {e / mean:.2%} | {s / mean:.2%} | {e / s - 1:+.0%} |")
    (LIVE / 'scoreboard.md').write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-7:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=['train', 'forecast', 'scoreboard'])
    {'train': train, 'forecast': forecast, 'scoreboard': scoreboard}[parser.parse_args().command]()


if __name__ == '__main__':
    main()
