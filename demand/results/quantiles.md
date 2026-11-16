Quantile forecasts, 2025-12-01 → 2026-10-30, issue time 12:00 D-1, monthly walk-forward.
cover = share of quarter-hours inside the interval (target 50% / 90%); width and pinball loss are relative to mean load (lower is better, pinball averaged over q = 0.05, 0.25, 0.75, 0.95).

| region | method | cover 50% | width 50% | cover 90% | width 90% | pinball | nMAE |
|---|---|---|---|---|---|---|---|
| DE | empirical (ensemble + 90-day residuals) | 49.52% | 4.39% | 89.36% | 10.52% | 0.68% | 2.55% |
| DE | LightGBM quantiles | 45.45% | 4.53% | 87.84% | 13.15% | 0.78% | 2.66% |
| DE | LightGBM quantiles on ensemble | 44.70% | 4.53% | 87.52% | 13.15% | 0.79% | 2.55% |
| 50Hertz | empirical (ensemble + 90-day residuals) | 51.94% | 9.50% | 90.14% | 22.27% | 1.41% | 5.28% |
| 50Hertz | LightGBM quantiles | 48.57% | 8.99% | 86.12% | 21.72% | 1.48% | 5.43% |
| 50Hertz | LightGBM quantiles on ensemble | 49.28% | 8.99% | 86.65% | 21.72% | 1.46% | 5.28% |
| Amprion | empirical (ensemble + 90-day residuals) | 46.96% | 5.63% | 86.29% | 14.46% | 1.02% | 3.65% |
| Amprion | LightGBM quantiles | 46.81% | 5.92% | 86.43% | 16.01% | 1.09% | 3.79% |
| Amprion | LightGBM quantiles on ensemble | 46.76% | 5.92% | 86.13% | 16.01% | 1.09% | 3.65% |
| TenneT | empirical (ensemble + 90-day residuals) | 46.72% | 4.80% | 86.15% | 11.48% | 0.83% | 3.09% |
| TenneT | LightGBM quantiles | 43.53% | 4.78% | 84.64% | 13.81% | 0.94% | 3.19% |
| TenneT | LightGBM quantiles on ensemble | 43.73% | 4.78% | 84.27% | 13.81% | 0.93% | 3.09% |
| TransnetBW | empirical (ensemble + 90-day residuals) | 49.13% | 4.57% | 88.68% | 12.11% | 0.81% | 2.93% |
| TransnetBW | LightGBM quantiles | 49.68% | 5.18% | 89.84% | 16.18% | 0.92% | 3.10% |
| TransnetBW | LightGBM quantiles on ensemble | 49.48% | 5.18% | 90.71% | 16.18% | 0.91% | 2.93% |
