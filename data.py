import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "kwh"]

def generate_sample_data(days: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate reproducible synthetic hourly energy readings with daily/weekly patterns and occasional spikes."""
    rng = np.random.default_rng(seed)
    periods = days * 24
    timestamps = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=periods, freq="h")
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()

    # Base load + morning and evening activity + lower weekend daytime activity.
    base = 0.22
    morning = 0.30 * np.exp(-0.5 * ((hour - 8) / 2.0) ** 2)
    evening = 0.55 * np.exp(-0.5 * ((hour - 19) / 3.0) ** 2)
    daytime = 0.16 * np.exp(-0.5 * ((hour - 13) / 3.5) ** 2)
    weekend_factor = np.where(weekday >= 5, 1.12, 1.0)
    trend = np.linspace(0, 0.04, periods)
    noise = rng.normal(0, 0.055, periods)
    values = (base + morning + evening + daytime) * weekend_factor + trend + noise
    values = np.clip(values, 0.05, None)

    # Inject a small number of synthetic spikes to test anomaly detection.
    spike_count = max(2, periods // 180)
    spike_idx = rng.choice(np.arange(24, periods), size=min(spike_count, max(1, periods - 24)), replace=False)
    values[spike_idx] += rng.uniform(0.7, 1.5, len(spike_idx))

    return pd.DataFrame({"timestamp": timestamps, "kwh": np.round(values, 3)})

def load_uploaded_data(file_obj) -> pd.DataFrame:
    df = pd.read_csv(file_obj)
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Accept common alternatives.
    if "timestamp" not in df.columns:
        for alt in ["datetime", "date_time", "time", "date"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "timestamp"})
                break
    if "kwh" not in df.columns:
        for alt in ["energy_kwh", "consumption", "usage_kwh", "energy", "usage"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "kwh"})
                break
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns not found: {missing}. Expected timestamp and kwh.")
    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS).sort_values("timestamp")
    return df.reset_index(drop=True)

def validate_readings(df: pd.DataFrame):
    errors = []
    if df.empty:
        errors.append("No valid rows found.")
    if df["timestamp"].isna().any():
        errors.append("Some timestamps are invalid.")
    if df["kwh"].isna().any():
        errors.append("Some energy values are not numeric.")
    if (df["kwh"] < 0).any():
        errors.append("Energy readings cannot be negative.")
    if df["timestamp"].duplicated().any():
        errors.append("Duplicate timestamps found; remove duplicates before upload.")
    return errors
