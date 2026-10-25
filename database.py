import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "ecopulse.db"

@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                timestamp TEXT PRIMARY KEY,
                kwh REAL NOT NULL,
                source TEXT DEFAULT 'local'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                forecast_json TEXT NOT NULL,
                anomalies_json TEXT NOT NULL
            )
        """)

def save_readings(df: pd.DataFrame):
    if df.empty:
        return
    clean = df[["timestamp", "kwh"]].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO readings(timestamp, kwh, source) VALUES (?, ?, ?)",
            [(row.timestamp, float(row.kwh), "sample_or_upload") for row in clean.itertuples(index=False)]
        )

def load_readings() -> pd.DataFrame:
    with _connect() as conn:
        df = pd.read_sql_query("SELECT timestamp, kwh FROM readings ORDER BY timestamp", conn)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def save_analysis(summary: dict, forecast: pd.DataFrame, anomalies: pd.DataFrame):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO analyses(created_at, summary_json, forecast_json, anomalies_json) VALUES (?, ?, ?, ?)",
            (
                pd.Timestamp.now().isoformat(timespec="seconds"),
                json.dumps(summary),
                forecast.to_json(orient="records", date_format="iso"),
                anomalies.to_json(orient="records", date_format="iso")
            )
        )

def load_analyses() -> pd.DataFrame:
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT id, created_at, summary_json FROM analyses ORDER BY id DESC LIMIT 100", conn
        )
    if not df.empty:
        df["summary"] = df["summary_json"].apply(lambda s: json.loads(s))
        expanded = pd.json_normalize(df["summary"])
        df = pd.concat([df.drop(columns=["summary_json", "summary"]), expanded], axis=1)
    return df
