import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(x["timestamp"])
    x["hour"] = ts.dt.hour
    x["dayofweek"] = ts.dt.dayofweek
    x["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    x["lag_1"] = x["kwh"].shift(1)
    x["lag_24"] = x["kwh"].shift(24)
    x["rolling_mean_24"] = x["kwh"].shift(1).rolling(24, min_periods=6).mean()
    return x

def forecast_next_day(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    clean = df[["timestamp", "kwh"]].copy().dropna().sort_values("timestamp")
    if len(clean) < 48:
        return pd.DataFrame(columns=["timestamp", "predicted_kwh"]), {"mae": None, "rmse": None, "model": "Insufficient data"}
    features = _feature_frame(clean)
    feature_cols = ["hour", "dayofweek", "is_weekend", "lag_1", "lag_24", "rolling_mean_24"]
    train = features.dropna(subset=feature_cols + ["kwh"])
    if len(train) < 24:
        return pd.DataFrame(columns=["timestamp", "predicted_kwh"]), {"mae": None, "rmse": None, "model": "Insufficient features"}

    # Time-ordered holdout to avoid leakage in evaluation.
    split = max(int(len(train) * 0.8), len(train) - 24)
    split = min(split, len(train) - 1)
    tr, te = train.iloc[:split], train.iloc[split:]
    model_eval = RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model_eval.fit(tr[feature_cols], tr["kwh"])
    pred_eval = model_eval.predict(te[feature_cols])
    mae = float(mean_absolute_error(te["kwh"], pred_eval))
    rmse = float(np.sqrt(mean_squared_error(te["kwh"], pred_eval)))

    model = RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(train[feature_cols], train["kwh"])

    # Recursive 24-hour forecast: future lags use earlier predictions.
    history = clean.copy()
    future_times = pd.date_range(start=clean["timestamp"].max() + pd.Timedelta(hours=1), periods=24, freq="h")
    preds = []
    for ts in future_times:
        vals = history["kwh"].to_numpy()
        lag1 = float(vals[-1])
        lag24 = float(vals[-24]) if len(vals) >= 24 else float(np.mean(vals))
        roll = float(np.mean(vals[-24:])) if len(vals) >= 6 else float(np.mean(vals))
        row = pd.DataFrame([{
            "hour": ts.hour,
            "dayofweek": ts.dayofweek,
            "is_weekend": int(ts.dayofweek >= 5),
            "lag_1": lag1,
            "lag_24": lag24,
            "rolling_mean_24": roll
        }])
        p = max(0.0, float(model.predict(row[feature_cols])[0]))
        preds.append(p)
        history = pd.concat([history, pd.DataFrame([{"timestamp": ts, "kwh": p}])], ignore_index=True)
    forecast = pd.DataFrame({"timestamp": future_times, "predicted_kwh": np.round(preds, 3)})
    return forecast, {"mae": round(mae, 4), "rmse": round(rmse, 4), "model": "Random Forest Regression"}

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    x = df[["timestamp", "kwh"]].copy().dropna().sort_values("timestamp").reset_index(drop=True)
    if len(x) < 24:
        x["is_anomaly"] = False
        x["anomaly_score"] = 0.0
        return x
    ts = pd.to_datetime(x["timestamp"])
    x["hour"] = ts.dt.hour
    x["dayofweek"] = ts.dt.dayofweek
    x["rolling_mean_24"] = x["kwh"].shift(1).rolling(24, min_periods=6).mean()
    x["deviation"] = (x["kwh"] - x["rolling_mean_24"]).fillna(0)
    features = x[["kwh", "hour", "dayofweek", "deviation"]].replace([np.inf, -np.inf], 0).fillna(0)
    model = IsolationForest(n_estimators=150, contamination=0.025, random_state=42)
    labels = model.fit_predict(features)
    x["is_anomaly"] = labels == -1
    x["anomaly_score"] = np.round(-model.decision_function(features), 4)
    return x[["timestamp", "kwh", "is_anomaly", "anomaly_score"]]

def analyze_energy(df: pd.DataFrame) -> dict:
    forecast, metrics = forecast_next_day(df)
    anomalies = detect_anomalies(df)
    summary = {
        "rows": int(len(df)),
        "mean_hourly_kwh": round(float(df["kwh"].mean()), 4) if len(df) else 0,
        "max_hourly_kwh": round(float(df["kwh"].max()), 4) if len(df) else 0,
        "anomaly_count": int(anomalies["is_anomaly"].sum()) if len(anomalies) else 0,
        "forecast_mae": metrics["mae"],
        "forecast_rmse": metrics["rmse"],
        "model": metrics["model"]
    }
    return {"forecast": forecast, "anomalies": anomalies, "summary": summary, "metrics": metrics}
