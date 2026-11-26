# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

19 day(s) scored, 2026-11-01 → 2026-11-24.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 1.98% | 3.58% | -45% |
| 50Hertz | 4.26% | 7.19% | -41% |
| Amprion | 2.56% | 4.58% | -44% |
| TenneT | 3.26% | 5.37% | -39% |
| TransnetBW | 3.01% | 3.79% | -20% |

Uncertainty bands, share of quarter-hours inside (target 50% / 90%):

| region | days | 50% band | 90% band |
|---|---|---|---|
| DE | 8 | 52% | 93% |
| 50Hertz | 8 | 37% | 80% |
| Amprion | 8 | 69% | 99% |
| TenneT | 8 | 44% | 85% |
| TransnetBW | 8 | 42% | 87% |
