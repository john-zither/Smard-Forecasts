# Beating SMARD's 15-minute load forecasts

[SMARD](https://www.smard.de) (Bundesnetzagentur) publishes a day-ahead
forecast of grid load for Germany and each of the four transmission
regions. This project forecasts the same quantity at the same time **without
using SMARD's forecast**, and compares the two on the same quarter-hours.

## Which forecast to beat

An audit of SMARD's day-ahead forecasts, Nov 2025 – Oct 2026, against
"same quarter-hour last week" (a forecast anyone can make):

| SMARD forecast | Error (nMAE) | Better than last week by |
|---|---|---|
| Wind onshore, DE | 9.8% | 87% |
| Solar, DE | 6.8% | 79% |
| Load, DE | 3.8% | 27% |
| Load, Amprion | 4.4% | 29% |
| **Load, 50Hertz** | 8.2% | **−3%** |
| **Load, TenneT** | 6.4% | **−7%** |
| **Load, TransnetBW** | 7.1% | **−3%** |

The regional load forecasts are SMARD's weakest: three of four are no
better than last week's load.

## Results

Forecasts issued at 12:00 the day before, monthly walk-forward (each month
forecast by models trained only on data up to two days before it), all
34,000+ quarter-hours per region from 2025-11-01 to 2026-10-30. nMAE = mean
absolute error / mean load.

| Region | SMARD | Best simple fix | **This model** | vs SMARD |
|---|---|---|---|---|
| DE | 3.79% | 3.35% | **2.54%** | −33% |
| 50Hertz | 8.13% | 6.80% | **5.33%** | −35% |
| Amprion | 4.40% | 4.05% | **3.54%** | −19% |
| TenneT | 6.39% | 5.33% | **2.99%** | −53% |
| TransnetBW | 7.07% | 4.38% | **2.87%** | −59% |

"Best simple fix" is the better of two trivial corrections of SMARD's own
forecast: subtracting its trailing 28-day per-quarter-hour bias, or
averaging it with last week's load. Any useful model has to beat those,
not just SMARD. The model beats SMARD in 58 of 60 region-months
(`results/ensemble.md`); the exceptions are Amprion in Aug–Sep 2026.

## How it works

**Issue-time rules** (`demand/baselines.py`). At noon on D-1, actual load is
published up to about 10:15, so features may use full days up to D-2 and
D-1's morning. A unit test changes load after 10:15 on D-1 and on day D and
checks the features for D don't move. Weather is Open-Meteo's archive of
forecasts *as issued two days earlier* (`*_previous_day2`), not observed
weather, so the backtest can't see the weather that actually happened.
SMARD's first-published actuals match today's values to within ~0.3%, so
later revisions don't flatter the backtest.

**Features** (`demand/features.py`). Load is scaled by each region's mean
over the 28 days ending D-2, so one model serves regions whose load differs
eightfold. Inputs: load 2, 7, 14, 21 and 364 days back; how yesterday
morning compares with the same morning a week earlier; forecast
temperature, solar radiation (behind-the-meter PV lowers grid load) and
cloud cover; holidays weighted by each state's share of the region; calendar.

**Models**, both trained on 2016 onwards:

* `demand/lgbm.py`: LightGBM with an L1 objective, one row per
  (day, region, quarter-hour), on CPU.
* `demand/nn.py`: a network that forecasts all five regions' 96
  quarter-hours at once: dilated convolutions along the day, with a learned
  mixing across regions after each block. Trained on the homelab cluster's
  shared RTX 2060 as a Kubernetes Job (3 GPU slots, about 47 minutes for 12
  folds × 3 seeds; `make demand-train` in `../../homelab`).
* `demand/ensemble.py`: the average of the two, the forecast reported above.
  The two models are about equally accurate alone; averaging gains 2–5%.

## Caveats

* One year of out-of-sample test. SMARD's TransnetBW forecast improved
  sharply from Aug 2026 (≈11% → ≈4%), so the gap there is shrinking.
* SMARD may publish its forecast earlier than noon; it was already out at
  11:37 on 2026-10-31. If it's published hours earlier, this model has a
  slightly fresher view of D-1's load.
* No live record yet; a daily noon run that logs forecasts before the
  actuals exist would remove any doubt about the backtest.

## Run

```bash
uv sync --extra dev
uv run python -m demand.data        # SMARD since 2015 (~2.5 min) + weather forecasts since 2021
uv run python -m demand.evaluate    # baselines -> results/baselines.md
uv run python -m demand.lgbm        # ~20 min on CPU -> results/lgbm.md
make -C ../../homelab demand-train  # neural model on the GPU -> results/nn.md
uv run python -m demand.ensemble    # -> results/ensemble.md
uv run pytest
```

Data: SMARD (Bundesnetzagentur, CC BY 4.0); weather: Open-Meteo (CC BY 4.0).
