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
def fetch_smard_series(filter_id, region, existing_data=None):
    """Deep fetch: grabs last 8 chunks to ensure 3+ days of history."""
    try:
        url_idx = f"https://www.smard.de/app/chart_data/{filter_id}/{region}/index_quarterhour.json"
        indices = requests.get(url_idx, timeout=30).json()["timestamps"]

        all_data = []
        # Grab last 8 chunks to cover historical backfilling needs
        for ts in indices[-8:]:
            url = f"https://www.smard.de/app/chart_data/{filter_id}/{region}/{filter_id}_{region}_quarterhour_{ts}.json"
            res = requests.get(url, timeout=30).json()
            if "series" in res:
                all_data.extend(res["series"])

        df = pd.DataFrame(all_data, columns=["timestamp", "value"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Europe/Berlin")
        
        # Deduplicate and resample
        series = df.set_index("datetime")["value"].resample("15min").mean()
        
        if existing_data is not None:
            series = pd.concat([existing_data, series])
            series = series[~series.index.duplicated(keep="last")].sort_index()
            
        return series.dropna()
    except Exception as e:
        print(f"  Error fetching SMARD {filter_id}: {e}")
        return existing_data if existing_data is not None else pd.Series()

def get_feature_vector(target_ts, issue_ts, actuals, smard_forecast, required_cols):
    """Strict feature builder - returns None if ANY required data is NaN."""
    safe_deadline = issue_ts - timedelta(minutes=75)
    row = {}
    
    try:
        row["qh"] = target_ts.hour * 4 + target_ts.minute // 15
        row["dow"] = target_ts.dayofweek
        row["is_holiday"] = int(target_ts in holidays.Germany())

        # Inputs for bias
        val_act = actuals.asof(safe_deadline)
        val_sm = smard_forecast.asof(safe_deadline)
        if pd.isna(val_act) or pd.isna(val_sm): return None
        row["smard_bias_at_lag"] = val_act - val_sm

        # Rolling history
        hist = actuals[actuals.index <= safe_deadline].tail(96)
        if len(hist) < 96: return None
        row["rolling_24h_at_lag"] = hist.mean()

        # Day lags
        for d in range(1, 8):
            v = actuals.asof(safe_deadline - timedelta(days=d-1))
            if pd.isna(v): return None
            row[f"lag_now_day_{d}"] = v
        
        for d in range(2, 8):
            v = actuals.asof(target_ts - timedelta(days=d))
            if pd.isna(v): return None
            row[f"lag_target_day_{d}"] = v

        # Baseline
        base = smard_forecast.asof(target_ts - timedelta(days=1))
        if pd.isna(base): return None
        row["smard_today_baseline"] = base

        if "forecast_smard" in required_cols:
            v = smard_forecast.asof(target_ts)
            if pd.isna(v): return None
            row["forecast_smard"] = v

        return pd.DataFrame([row]).reindex(columns=required_cols).astype(float)
    except:
        return None

# =========================
# Main
# =========================
def main():
    if DB_PATH.exists():
        db = pd.read_csv(DB_PATH)
        db["datetime"] = pd.to_datetime(db["datetime"], utc=True).dt.tz_convert("Europe/Berlin")
    else:
        db = pd.DataFrame(columns=["datetime", "time_calculated", "model", "value", "region"])

    now_berlin = datetime.now().astimezone(pd.Timestamp.now(tz="Europe/Berlin").tz)
    # Range: 3 days ago through tomorrow
    target_dates = [(now_berlin + timedelta(days=i)).date() for i in range(-3, 2)]

    for tso_name, tso_id in REGIONS.items():
        print(f"Processing {tso_name}...")
        
        # 1. Fetch data
        actuals = fetch_smard_series(410, tso_id)
        smard_f = fetch_smard_series(411, tso_id)

        # 2. SAVE RAW DATA IMMEDIATELY (The "Grab")
        for label, series in [("actual", actuals), ("smard_forecast", smard_f)]:
            if series.empty: continue
            new_rows = series.reset_index().rename(columns={"index": "datetime", 0: "value"})
            new_rows["model"], new_rows["region"], new_rows["time_calculated"] = label, tso_name, datetime.now()
            db = pd.concat([db, new_rows]).drop_duplicates(subset=["datetime", "model", "region"], keep="last")

        # 3. Model Loop
        for target_date in target_dates:
            target_range = pd.date_range(start=str(target_date), periods=96, freq="15min", tz="Europe/Berlin")
            
            for h in MODELS:
                m_label = str(h)
                # Skip if this specific model/date combo is already 100% full
                existing_count = len(db[(db["datetime"].dt.date == target_date) & (db["model"] == m_label) & (db["region"] == tso_name)])
                if existing_count >= 96: continue

                model_path = OUTPUT_DIR / tso_name / f"{h:02d}" / "best_model.pkl"
                if not model_path.exists(): continue
                with open(model_path, "rb") as f: pkg = pickle.load(f)

                issue_hour = 0 if h == 24 else h
                issue_ts = pd.Timestamp(target_date - timedelta(days=1), tz="Europe/Berlin").replace(hour=issue_hour)
                
                # Logic: Don't run models whose issue time hasn't happened yet
                if issue_ts > now_berlin: continue

                preds, valid_times = [], []
                for ts in target_range:
                    feats = get_feature_vector(ts, issue_ts, actuals, smard_f, pkg["cols"])
                    if feats is not None:
                        preds.append(pkg["model"].predict(feats)[0])
                        valid_times.append(ts)

                if preds:
                    batch = pd.DataFrame({"datetime": valid_times, "value": preds})
                    batch["time_calculated"], batch["model"], batch["region"] = datetime.now(), m_label, tso_name
                    db = pd.concat([db, batch]).drop_duplicates(subset=["datetime", "model", "region"], keep="last")

    db.to_csv(DB_PATH, index=False)
    print("Tasks complete.")

if __name__ == "__main__":
    main()