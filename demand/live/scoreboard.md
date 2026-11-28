# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

21 day(s) scored, 2026-11-01 → 2026-11-27.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 1.89% | 3.57% | -47% |
| 50Hertz | 4.10% | 7.15% | -43% |
| Amprion | 2.49% | 4.71% | -47% |
| TenneT | 3.37% | 5.31% | -37% |
| TransnetBW | 2.97% | 3.85% | -23% |

Uncertainty bands, share of quarter-hours inside (target 50% / 90%):

| region | days | 50% band | 90% band |
|---|---|---|---|
| DE | 10 | 57% | 93% |
| 50Hertz | 10 | 41% | 82% |
| Amprion | 10 | 73% | 99% |
| TenneT | 10 | 43% | 83% |
| TransnetBW | 10 | 41% | 88% |
