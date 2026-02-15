import os
import pickle
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timedelta
import holidays
import warnings

warnings.filterwarnings("ignore")

# --- CONFIG ---
REGIONS = {
    "DE": "DE",
    "50Hertz": "50Hertz",
    "Amprion": "Amprion",
    "TenneT": "TenneT",
    "TransnetBW": "TransnetBW",
}
OUTPUT_DIR = Path("output")
DB_PATH = Path("forecast_database.csv")
MODELS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


# =========================
# Helpers
# =========================
def is_series_fresh(series, now_ts, max_age_minutes=120):
    """Check if latest datapoint is recent enough."""
    if series is None or series.empty:
        return False
    last_ts = series.index.max()
    return last_ts >= (now_ts - timedelta(minutes=max_age_minutes))


def fetch_smard_series(filter_id, region, existing_data=None):
    """Fetch SMARD data with cache awareness."""
    now_berlin = datetime.now().astimezone(
        pd.Timestamp.now(tz="Europe/Berlin").tz
    )

    # Use cache if fresh
    if existing_data is not None and not existing_data.empty:
        last_ts = existing_data.index.max()
        if last_ts >= (now_berlin - timedelta(minutes=15)):
            print(f"  Using cached data for {region} ({filter_id})")
            return existing_data

    try:
        url_idx = (
            f"https://www.smard.de/app/chart_data/"
            f"{filter_id}/{region}/index_quarterhour.json"
        )
        indices = requests.get(url_idx, timeout=30).json()["timestamps"]

        all_data = []
        for ts in indices[-3:]:
            url = (
                f"https://www.smard.de/app/chart_data/"
                f"{filter_id}/{region}/"
                f"{filter_id}_{region}_quarterhour_{ts}.json"
            )
            all_data.extend(requests.get(url, timeout=30).json()["series"])

        df = pd.DataFrame(all_data, columns=["timestamp", "value"])
        df["datetime"] = (
            pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            .dt.tz_convert("Europe/Berlin")
        )
        new_series = (
            df.set_index("datetime")["value"].resample("15min").mean()
        )

        if existing_data is not None:
            combined = pd.concat([existing_data, new_series])
            combined = combined[
                ~combined.index.duplicated(keep="last")
            ].sort_index()
            series = combined
        else:
            series = new_series

        if filter_id == 410:
            series = series[
                series.index <= (now_berlin - timedelta(minutes=15))
            ].dropna()
        else:
            series = series.dropna()

        return series

    except Exception as e:
        print(f"  Error fetching SMARD {filter_id}: {e}")
        return existing_data if existing_data is not None else pd.Series()


# =========================
# Feature Builder (SAFE)
# =========================
def get_feature_vector(target_ts, issue_ts, actuals, smard_forecast, required_cols):
    """Strict feature builder — returns None if data insufficient."""

    # HARD GUARD
    if actuals.empty or smard_forecast.empty:
        return None

    safe_deadline = issue_ts - timedelta(minutes=75)

    # Detect stale SMARD (key fix)
    last_smard_ts = smard_forecast.index.max()
    if last_smard_ts < safe_deadline - timedelta(hours=2):
        return None

    row = {}
    row["qh"] = target_ts.hour * 4 + target_ts.minute // 15
    row["dow"] = target_ts.dayofweek
    row["is_holiday"] = int(target_ts in holidays.Germany())

    val_actual_at_lag = actuals.asof(safe_deadline)
    val_smard_at_lag = smard_forecast.asof(safe_deadline)

    # Require valid bias inputs
    if pd.isna(val_actual_at_lag) or pd.isna(val_smard_at_lag):
        return None

    row["smard_bias_at_lag"] = val_actual_at_lag - val_smard_at_lag

    history_24h = actuals[actuals.index <= safe_deadline].tail(96)
    if history_24h.empty:
        return None
    row["rolling_24h_at_lag"] = history_24h.mean()

    for d in range(1, 8):
        val = actuals.asof(safe_deadline - timedelta(days=d - 1))
        if pd.isna(val):
            return None
        row[f"lag_now_day_{d}"] = val

    for d in range(2, 8):
        val = actuals.asof(target_ts - timedelta(days=d))
        if pd.isna(val):
            return None
        row[f"lag_target_day_{d}"] = val

    baseline = smard_forecast.asof(target_ts - timedelta(days=1))
    if pd.isna(baseline):
        return None
    row["smard_today_baseline"] = baseline

    if "forecast_smard" in required_cols:
        val = smard_forecast.asof(target_ts)
        if pd.isna(val):
            return None
        row["forecast_smard"] = val

    return (
        pd.DataFrame([row])
        .reindex(columns=required_cols)
        .astype(float)
    )


# =========================
# Main
# =========================
def main():
    # Load DB
    if DB_PATH.exists():
        db = pd.read_csv(DB_PATH)
        db["datetime"] = (
            pd.to_datetime(db["datetime"], utc=True)
            .dt.tz_convert("Europe/Berlin")
        )
        db["time_calculated"] = pd.to_datetime(db["time_calculated"])
    else:
        db = pd.DataFrame(
            columns=["datetime", "time_calculated", "model", "value", "region"]
        )

    now_berlin = datetime.now().astimezone(
        pd.Timestamp.now(tz="Europe/Berlin").tz
    )
    calc_time = datetime.now()
    target_days = [now_berlin.date(), now_berlin.date() + timedelta(days=1)]

    for tso_name, tso_id in REGIONS.items():
        print(f"Processing {tso_name}...")

        cached_actuals = db[
            (db["region"] == tso_name) & (db["model"] == "actual")
        ].set_index("datetime")["value"]

        cached_smard = db[
            (db["region"] == tso_name) & (db["model"] == "smard_forecast")
        ].set_index("datetime")["value"]

        actuals = fetch_smard_series(410, tso_id, cached_actuals)
        smard_f = fetch_smard_series(411, tso_id, cached_smard)

        # 🚨 HARD DATA QUALITY GATE
        actuals_ok = is_series_fresh(actuals, now_berlin, 120)
        smard_ok = is_series_fresh(smard_f, now_berlin, 120)

        if not actuals_ok or not smard_ok:
            print(f"  ⚠️ Skipping {tso_name} — data not fresh")
            continue

        # Update benchmarks
        for label, series in [("actual", actuals), ("smard_forecast", smard_f)]:
            if series.empty:
                continue
            new_rows = series.reset_index().rename(
                columns={"index": "datetime", 0: "value"}
            )
            new_rows["time_calculated"] = calc_time
            new_rows["model"] = label
            new_rows["region"] = tso_name
            db = pd.concat([db, new_rows]).drop_duplicates(
                subset=["datetime", "model", "region"], keep="last"
            )

        # Run models
        for target_date in target_days:
            target_range = pd.date_range(
                start=str(target_date),
                periods=96,
                freq="15min",
                tz="Europe/Berlin",
            )

            run_models = (
                [m for m in MODELS if m <= now_berlin.hour]
                if target_date > now_berlin.date()
                else MODELS
            )

            for h in run_models:
                m_label = str(h)

                if not db[
                    (db["datetime"].dt.date == target_date)
                    & (db["model"] == m_label)
                    & (db["region"] == tso_name)
                ].empty:
                    continue

                model_path = OUTPUT_DIR / tso_name / f"{h:02d}" / "best_model.pkl"
                if not model_path.exists():
                    continue

                with open(model_path, "rb") as f:
                    pkg = pickle.load(f)

                issue_ts = pd.Timestamp(
                    target_date - timedelta(days=1),
                    tz="Europe/Berlin",
                ).replace(hour=0 if h == 24 else h)

                print(f"  -> Generating {m_label} forecast for {target_date}")

                preds = []
                for ts in target_range:
                    feats = get_feature_vector(
                        ts, issue_ts, actuals, smard_f, pkg["cols"]
                    )
                    if feats is None:
                        preds = []
                        print(f"    ⚠️ Skipping {m_label} — insufficient data")
                        break
                    preds.append(pkg["model"].predict(feats)[0])

                if not preds:
                    continue

                batch = pd.DataFrame(
                    {
                        "datetime": target_range,
                        "time_calculated": calc_time,
                        "model": m_label,
                        "value": preds,
                        "region": tso_name,
                    }
                )
                db = pd.concat([db, batch])

    db.to_csv(DB_PATH, index=False)
    print("All tasks complete.")


if __name__ == "__main__":
    main()
