"""What is known at the noon D-1 issue time, and the simple forecasts built from it.

The forecast for delivery day D is issued at 12:00 on D-1. By then:
  * SMARD's own forecast for all of D is published (it appears in the
    morning; checked live on 2026-10-31 at 11:37);
  * actual load is published with a lag of about an hour, so D-1 is known
    up to 10:15 (MORNING_SLOTS quarter-hours) and every earlier day fully.

Simple forecasts any model has to beat:
  smard        SMARD's published forecast
  last_week    actual load of D-7
  smard_debiased  SMARD minus its mean error at that quarter-hour over the
               28 days ending D-2 (the last complete day)
  blend        mean of SMARD and last_week
"""

import numpy as np
import pandas as pd

MORNING_SLOTS = 42         # 00:00 - 10:15 of D-1 are published by noon
BIAS_DAYS = 28


def lag(m: pd.DataFrame, days: int) -> pd.DataFrame:
    """Row for day D holds the values of day D-days."""
    return m.shift(days, freq='D').reindex(m.index)


def trailing_bias(actual: pd.DataFrame, smard: pd.DataFrame, days: int = BIAS_DAYS) -> pd.DataFrame:
    """Per-slot mean of (SMARD - actual) over days D-2-days+1 .. D-2."""
    err = (smard - actual).reindex(smard.index)
    return lag(err.rolling(days, min_periods=days // 2).mean(), 2)


def baselines(actual: pd.DataFrame, smard: pd.DataFrame) -> dict[str, pd.DataFrame]:
    idx = smard.index
    d7 = lag(actual.reindex(idx), 7)
    return {
        'smard': smard,
        'last_week': d7,
        'smard_debiased': smard - trailing_bias(actual, smard),
        'blend': (smard + d7) / 2,
    }
