# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

6 day(s) scored, 2026-11-01 → 2026-11-06.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 2.14% | 4.19% | -49% |
| 50Hertz | 3.87% | 6.67% | -42% |
| Amprion | 3.28% | 5.36% | -39% |
| TenneT | 2.80% | 4.92% | -43% |
| TransnetBW | 3.64% | 4.65% | -22% |
