import pandas as pd

def build_recommendations(df: pd.DataFrame, anomalies: pd.DataFrame, forecast: pd.DataFrame):
    items = []
    if df.empty:
        return [{"title": "Add energy readings", "detail": "Generate sample data or upload a CSV to receive analysis.", "priority": "High", "evidence": "No readings available"}]

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    night = data[data["timestamp"].dt.hour.isin([0, 1, 2, 3, 4, 5])]
    overall = float(data["kwh"].mean())
    night_mean = float(night["kwh"].mean()) if not night.empty else 0.0

    if not night.empty and night_mean > overall * 0.85:
        items.append({
            "title": "Review overnight base load",
            "detail": "Overnight average usage is relatively close to the overall hourly average. Check whether always-on devices, HVAC, lighting, or equipment schedules explain this pattern.",
            "priority": "Medium",
            "evidence": f"Night average {night_mean:.3f} kWh/hour vs overall average {overall:.3f} kWh/hour"
        })
    else:
        items.append({
            "title": "Track overnight base load",
            "detail": "Compare overnight readings across several days. A persistent unexplained base load can be a useful starting point for an energy audit.",
            "priority": "Low",
            "evidence": f"Overall average {overall:.3f} kWh/hour"
        })

    if not anomalies.empty and anomalies["is_anomaly"].any():
        count = int(anomalies["is_anomaly"].sum())
        items.append({
            "title": "Investigate unusual readings",
            "detail": "Review the flagged timestamps and check occupancy, weather, appliance use, or meter data quality. An anomaly may be legitimate and does not by itself prove waste.",
            "priority": "High",
            "evidence": f"{count} readings flagged by Isolation Forest"
        })
    else:
        items.append({
            "title": "Continue anomaly monitoring",
            "detail": "No readings were flagged in this run. Keep monitoring because a quiet period does not guarantee that energy waste is absent.",
            "priority": "Low",
            "evidence": "No anomaly flags in current dataset"
        })

    if not forecast.empty:
        expected = float(forecast["predicted_kwh"].sum())
        historical_daily = data.set_index("timestamp")["kwh"].resample("D").sum()
        baseline = float(historical_daily.tail(7).mean()) if len(historical_daily) else expected
        if expected > baseline * 1.10:
            detail = "Predicted consumption is above the recent daily average. Review planned occupancy and equipment schedules before peak usage."
            priority = "Medium"
        else:
            detail = "Use the forecast to plan equipment schedules and compare actual consumption with expected usage after the day ends."
            priority = "Low"
        items.append({
            "title": "Plan for the next-day forecast",
            "detail": detail,
            "priority": priority,
            "evidence": f"Forecast {expected:.2f} kWh; recent 7-day daily average {baseline:.2f} kWh"
        })

    items.append({
        "title": "Measure savings before claiming them",
        "detail": "After making a change, compare similar days and account for occupancy and weather. This prototype does not directly measure appliance-level savings.",
        "priority": "Medium",
        "evidence": "Research and evaluation best practice"
    })
    return items
