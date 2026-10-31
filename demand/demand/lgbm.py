"""Walk-forward LightGBM demand forecast, built without SMARD's forecast.

For each test month, train on every day up to two days before the month
starts (the last complete day at the first forecast's issue time), early-stop
on the final 42 of those days, and forecast the month. The model predicts
load over its recent level (demand.features); the forecast is that times
the level.

    uv run python -m demand.lgbm
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from demand.data import REGIONS, SLOTS
from demand.evaluate import RESULTS, TEST_END, TEST_START, monthly, run, table
from demand.features import build, feature_columns

PARAMS = dict(objective='l1', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=6)
VAL_DAYS = 42


def walk_forward(df: pd.DataFrame, params: dict = PARAMS, label: str = 'lgbm') -> pd.Series:
    """Out-of-sample predicted scaled load for every test row, indexed like df."""
    cols = feature_columns(df)
    pred = pd.Series(np.nan, index=df.index)
    for month in pd.date_range(TEST_START, TEST_END, freq='MS'):
        cutoff = month - pd.Timedelta('2D')
        train = df[(df.day <= cutoff) & df.target.notna()]
        fit, val = train[train.day <= cutoff - pd.Timedelta(f'{VAL_DAYS}D')], train[train.day > cutoff - pd.Timedelta(f'{VAL_DAYS}D')]
        test = df[(df.day >= month) & (df.day < month + pd.offsets.MonthBegin(1))]
        ds = lgb.Dataset(fit[cols], fit.target, categorical_feature=['region'])
        dv = lgb.Dataset(val[cols], val.target, reference=ds)
        model = lgb.train(params, ds, 2000, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        pred[test.index] = model.predict(test[cols], num_iteration=model.best_iteration)
        print(f"  {label} {month:%Y-%m}: {model.best_iteration} trees, {len(fit):,} training rows", flush=True)
    return pred


def to_matrices(df: pd.DataFrame, pred: pd.Series) -> dict[str, pd.DataFrame]:
    """{region: [day x 96] forecast in MWh per quarter-hour} from predicted scaled load."""
    out = {}
    for r, region in enumerate(REGIONS):
        part = df[(df.region == r) & pred.notna()]
        mwh = (pred[part.index] * part.scale).to_numpy().reshape(-1, SLOTS)
        out[region] = pd.DataFrame(mwh, index=pd.DatetimeIndex(part.day.iloc[::SLOTS]))
    return out


def main() -> None:
    df = build()
    print(f"{len(df):,} rows, {len(feature_columns(df))} features", flush=True)
    pred = walk_forward(df)
    models = {'lgbm': to_matrices(df, pred)}
    res = run(models)
    text = (f"nMAE (MAE / mean load), {TEST_START.date()} → {TEST_END.date()}, issue time 12:00 D-1, "
            f"monthly walk-forward\n\n" + table(res) + "\n\nBy month:\n\n" + monthly(models['lgbm'], 'lgbm'))
    print(text)
    (RESULTS / 'lgbm.md').write_text(text + '\n')
    res.to_csv(RESULTS / 'lgbm.csv', index=False)
    pd.concat(models['lgbm'], names=['region', 'day']).to_parquet(RESULTS / 'lgbm_pred.parquet')


if __name__ == '__main__':
    main()
