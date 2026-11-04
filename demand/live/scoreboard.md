# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

3 day(s) scored, 2026-11-01 → 2026-11-03.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 2.14% | 3.37% | -36% |
| 50Hertz | 3.23% | 6.99% | -54% |
| Amprion | 3.68% | 6.25% | -41% |
| TenneT | 3.65% | 4.69% | -22% |
| TransnetBW | 4.96% | 4.81% | +3% |
