# SMARD Demand Forecast

SMARD (Bundesnetzagentur) publishes a day-ahead forecast of German grid
load, nationally and for each of the four transmission regions. This
project finds the weakest of those forecasts and beats it **without using
SMARD's forecast as an input**, with a live daily record.

## Which forecast to beat

Compared with "same quarter-hour last week", which anyone can produce,
SMARD's wind and solar forecasts are 79–87% better, and its national load
forecast 27% better. Its **regional load forecasts** are its weakest: for
50Hertz, TenneT and TransnetBW they are no better than last week's load.

## Results

Forecasts issued at 12:00 the day before, monthly walk-forward over
2025-11-01 → 2026-10-30. Each month is forecast by models trained only on
data up to two days before it, with weather *as forecast at the time*.
Error is MAE as a share of mean load.

| Region | SMARD | This model | Error vs SMARD |
|---|---|---|---|
| Germany | 3.79% | **2.54%** | −33% |
| 50Hertz | 8.13% | **5.33%** | −35% |
| Amprion | 4.40% | **3.54%** | −19% |
| TenneT | 6.39% | **2.99%** | −53% |
| TransnetBW | 7.07% | **2.87%** | −59% |

The model is the average of a LightGBM model and a neural network that
forecasts all five regions at once, trained on a home Kubernetes cluster's
GPU. It beats SMARD in 58 of 60 region-months. Methods, baselines, tests
and caveats are in **[demand/](demand/README.md)**.

**Live record:** every day at 12:00 the model forecasts tomorrow, next to
SMARD's forecast as published at that moment, and commits it to
[demand/live/forecasts/](demand/live/forecasts/) before the day happens.
Scoreboard: **[john-zither.github.io/Smard-Forecasts](https://john-zither.github.io/Smard-Forecasts/)**
([as markdown](demand/live/scoreboard.md)).

## History

The first version (`fetch_and_forecast.py`, published hourly to
`forecast_database.csv` from February to August 2026) claimed
validation results 25% better than SMARD. Scored on its live forecasts it
wasn't: over 169 days its noon model was worse than SMARD in every region
except 50Hertz, and often worse than last week's load. Two things changed:
weather forecasts (including solar radiation, since rooftop PV lowers grid
load at midday) and an evaluation that only allows data published by the
issue time. That GitHub Action is no longer maintained.
