# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

20 day(s) scored, 2026-11-01 → 2026-11-26.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 1.91% | 3.56% | -46% |
| 50Hertz | 4.22% | 7.15% | -41% |
| Amprion | 2.55% | 4.55% | -44% |
| TenneT | 3.18% | 5.34% | -40% |
| TransnetBW | 3.03% | 3.75% | -19% |

Uncertainty bands, share of quarter-hours inside (target 50% / 90%):

| region | days | 50% band | 90% band |
|---|---|---|---|
| DE | 9 | 56% | 93% |
| 50Hertz | 9 | 37% | 80% |
| Amprion | 9 | 69% | 99% |
| TenneT | 9 | 47% | 86% |
| TransnetBW | 9 | 42% | 86% |
