# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

14 day(s) scored, 2026-11-01 → 2026-11-19.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 2.05% | 3.72% | -45% |
| 50Hertz | 4.19% | 7.37% | -43% |
| Amprion | 2.76% | 4.65% | -41% |
| TenneT | 3.40% | 5.41% | -37% |
| TransnetBW | 3.08% | 3.76% | -18% |

Uncertainty bands, share of quarter-hours inside (target 50% / 90%):

| region | days | 50% band | 90% band |
|---|---|---|---|
| DE | 3 | 48% | 92% |
| 50Hertz | 3 | 33% | 74% |
| Amprion | 3 | 70% | 99% |
| TenneT | 3 | 32% | 72% |
| TransnetBW | 3 | 49% | 89% |
