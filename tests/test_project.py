"""Smoke tests for EcoPulse AI core data and ML functions. Run: python -m unittest discover -s tests -v"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data import generate_sample_data, validate_readings
from src.analysis import analyze_energy
from src.recommendations import build_recommendations
import src.database as database


class EcoPulseSmokeTests(unittest.TestCase):
    def test_sample_data_shape_and_values(self):
        df = generate_sample_data(days=7, seed=42)
        self.assertEqual(len(df), 7 * 24)
        self.assertEqual(list(df.columns), ["timestamp", "kwh"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))
        self.assertTrue((df["kwh"] >= 0).all())
        self.assertEqual(validate_readings(df), [])

    def test_validation_catches_negative_and_duplicate(self):
        ts = pd.Timestamp("2026-01-01 00:00:00")
        df = pd.DataFrame({"timestamp": [ts, ts], "kwh": [0.2, -0.1]})
        errors = validate_readings(df)
        self.assertTrue(any("negative" in e.lower() for e in errors))
        self.assertTrue(any("duplicate" in e.lower() for e in errors))

    def test_analysis_returns_24_hour_forecast_and_anomalies(self):
        df = generate_sample_data(days=7, seed=7)
        result = analyze_energy(df)
        self.assertEqual(len(result["forecast"]), 24)
        self.assertIn("is_anomaly", result["anomalies"].columns)
        self.assertGreaterEqual(result["forecast"]["predicted_kwh"].min(), 0)
        self.assertIsNotNone(result["metrics"]["mae"])
        self.assertIsNotNone(result["metrics"]["rmse"])

    def test_recommendations_are_generated(self):
        df = generate_sample_data(days=7, seed=9)
        result = analyze_energy(df)
        items = build_recommendations(df, result["anomalies"], result["forecast"])
        self.assertGreaterEqual(len(items), 3)
        self.assertTrue(all("title" in item and "evidence" in item for item in items))

    def test_sqlite_save_and_reload(self):
        original_path = database.DB_PATH
        try:
            with tempfile.TemporaryDirectory() as tmp:
                database.DB_PATH = Path(tmp) / "test.db"
                database.init_db()
                df = generate_sample_data(days=2, seed=4)
                database.save_readings(df)
                loaded = database.load_readings()
                self.assertEqual(len(loaded), len(df))
                result = analyze_energy(generate_sample_data(days=7, seed=4))
                database.save_analysis(result["summary"], result["forecast"], result["anomalies"])
                history = database.load_analyses()
                self.assertEqual(len(history), 1)
        finally:
            database.DB_PATH = original_path


if __name__ == "__main__":
    unittest.main()
