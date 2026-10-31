"""Average the LightGBM and neural forecasts (both SMARD-free) and score the result.

    uv run python -m demand.ensemble     # after demand.lgbm and demand.nn have written results/
"""

import numpy as np
import pandas as pd

from demand.data import REGIONS
from demand.evaluate import RESULTS, TEST_END, TEST_START, monthly, run, table


def load_predictions() -> dict[str, dict[str, pd.DataFrame]]:
    lg = pd.read_parquet(RESULTS / 'lgbm_pred.parquet')
    nn = np.load(RESULTS / 'nn_pred.npz')
    days = pd.DatetimeIndex(nn['days'])
    out = {'lgbm': {}, 'nn': {}}
    for r, region in enumerate(REGIONS):
        m = lg.loc[region]
        m.columns = m.columns.astype(int)
        out['lgbm'][region] = m
        out['nn'][region] = pd.DataFrame(nn['pred'][:, r], index=days)
    out['ensemble'] = {region: (out['lgbm'][region] + out['nn'][region].reindex(out['lgbm'][region].index)) / 2
                       for region in REGIONS}
    return out


def main() -> None:
    models = load_predictions()
    res = run(models)
    text = (f"nMAE (MAE / mean load), {TEST_START.date()} → {TEST_END.date()}, issue time 12:00 D-1, "
            f"monthly walk-forward; no model sees SMARD's forecast\n\n" + table(res)
            + "\n\nEnsemble by month:\n\n" + monthly(models['ensemble'], 'ensemble'))
    print(text)
    (RESULTS / 'ensemble.md').write_text(text + '\n')
    res.to_csv(RESULTS / 'ensemble.csv', index=False)


if __name__ == '__main__':
    main()
