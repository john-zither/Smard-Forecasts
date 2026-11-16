# Why three of SMARD's four regional demand forecasts are poor

SMARD's day-ahead grid-load forecast for Germany is the sum of the four transmission
operators' (TSOs') own forecasts. Amprion's is decent (4.0% nMAE since 2025). 50Hertz
(8.2%), TransnetBW (7.1%) and TenneT (5.9%) are no better than "same quarter-hour last
week". The poor regions don't share one bug; each has its own failure. The numbers below
come from SMARD's published actuals (filter 410) and forecasts (filter 411), plus the
Open-Meteo weather already cached in `data/`. Reproduce them with
`uv run python -m demand.diagnose`.

## TransnetBW: a fixed +400 MW offset since October 2020

Until September 2020 TransnetBW's forecast was unbiased. In the week of 5 October 2020
it started running about 250–500 MW above the measured load, and it has stayed there
for six years: +6.0% (2021), +7.9%, +7.1%, +5.5%, +6.3%, +6.0% (2026).

The offset is almost flat. It is +300 to +500 MW in every 3-hour block and every month,
and on every weekday (+410 Mon–Thu, +320 at weekends). It doesn't react to weather.
Something changed in the forecast's or the measurement's scope (a load definition, a
grid connection, a forecasting tool), and the forecast has never been recalibrated.
That is why the trivial fix works so well: subtracting the forecast's own trailing
28-day per-slot bias takes TransnetBW from **7.1% to 4.3%**. Most of its error is
this one offset.

## 50Hertz: since 15 June 2025, Sundays are forecast like working days

Up to spring 2025, 50Hertz forecast Sunday daytime load at about 91–95% of
Saturday's, which matches reality. From **Sunday 15 June 2025** onward, every Sunday
forecast sits *above* Saturday's:

| daytime load, Sunday / Saturday | actual | 50Hertz forecast |
|---|---|---|
| 2023 – Q1 2025 | 0.90 – 0.97 | 0.91 – 0.98 |
| Q3 2025 – Q4 2026 | 0.93 – 0.96 | **1.10 – 1.21** |

So Sunday forecasts are too high by about 2 GW during the day, on a region that
averages 11.6 GW. Since the change, Sundays are 14% of days but **24% of 50Hertz's
absolute error** (15.6% nMAE on Sundays, 7.2% on other days). A one-day calendar shift
doesn't explain it: the Sunday forecast's shape is closest to the following Monday's
actuals. It looks like a day-type or holiday-calendar regression in 50Hertz's
forecasting system that has gone unnoticed for 17 months.

50Hertz also had a huge under-forecast before 2022 (−26% in 2015, about −10 to −20%
through 2021). That bias went away in 2022–23.

## Every TSO under-models rooftop solar, and TenneT and 50Hertz most of all

On a sunny day, midday grid load drops because rooftop and other behind-the-meter PV
covers demand locally. Regressing the 10–14h load on midday radiation (with temperature,
weekday and year controls) shows how much of that dip each forecast anticipates:

| region | actual load response, MW per W/m² | forecast's response | captured |
|---|---|---|---|
| 50Hertz | −3.4 | −0.5 | 14% |
| TenneT | −5.4 | −1.0 | 19% |
| TransnetBW | −2.2 | −0.6 | 30% |
| Amprion | −3.8 | −2.1 | 56% |

TenneT (Bavaria and Lower Saxony, the most rooftop PV in Germany) loses about 4.4 MW
of accuracy per W/m². Between a cloudy (~100 W/m²) and a sunny (~700 W/m²) midday
that's roughly a 2.5 GW swing the forecast doesn't see. It shows up as a seasonal
pattern: TenneT over-forecasts by 500–700 MW from April to July and under-forecasts by
about 900 MW in January and November. The cold response (MW per °C below 15 °C) is
close to right everywhere, so solar, not heating, is the weather gap. Amprion is the
only TSO whose forecast clearly follows the sun, which goes a long way to explaining
why it's the good one.

## Errors persist for days

After removing the 28-day bias, a day's mean error still correlates +0.4 to +0.6 with
the previous day's. The forecasts don't appear to correct themselves from yesterday's
measured load. That is the "runs lasting hours to days" structure from the February
talk. It is cheap to exploit: our model uses recent load and beats SMARD in 58 of 60
region-months.

## What it means

The aggregation story from the talk (good regional fits that cancel badly when summed)
doesn't hold for SMARD. The regional forecasts are individually poor, each for an
ordinary operational reason:

- **TransnetBW:** a scope change in 2020 that was never recalibrated.
- **50Hertz:** a broken Sunday profile since mid-2025.
- **TenneT and 50Hertz:** rooftop solar is largely ignored.
- **All three:** no correction from recent measured load.

None of these would survive a basic forecast-monitoring dashboard, and that's an
inference about the TSOs' process, not something the data proves. The data does show
that the defects are stable and cheap to correct, which is why a SMARD-free model
beats the published forecast by 20–50%.
