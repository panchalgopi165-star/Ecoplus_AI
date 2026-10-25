import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from src.data import generate_sample_data, load_uploaded_data, validate_readings
from src.database import init_db, save_readings, save_analysis, load_readings, load_analyses
from src.analysis import analyze_energy
from src.recommendations import build_recommendations

st.set_page_config(page_title="EcoPulse AI", page_icon="⚡", layout="wide")
init_db()

st.title("⚡ EcoPulse AI")
st.caption("AI-powered energy monitoring, anomaly detection and consumption forecasting")

with st.sidebar:
    st.header("Data controls")
    source = st.radio("Choose data source", ["Generate sample data", "Upload CSV"])
    if source == "Generate sample data":
        days = st.slider("Days of sample history", min_value=7, max_value=90, value=30)
        seed = st.number_input("Random seed", min_value=1, max_value=99999, value=42)
        if st.button("Generate / refresh sample data", type="primary"):
            df = generate_sample_data(days=int(days), seed=int(seed))
            save_readings(df)
            st.session_state["active_data"] = df
            st.success(f"Generated {len(df)} hourly readings.")
    else:
        uploaded = st.file_uploader("Upload energy readings CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = load_uploaded_data(uploaded)
                errors = validate_readings(df)
                if errors:
                    st.error("CSV validation failed: " + "; ".join(errors))
                else:
                    save_readings(df)
                    st.session_state["active_data"] = df
                    st.success(f"Loaded {len(df)} readings.")
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")

    st.divider()
    st.markdown("**CSV format**")
    st.code("timestamp, kwh\n2026-01-01 00:00:00, 0.42", language="text")
    st.caption("Use hourly interval readings in kWh. Sample data is synthetic.")

if "active_data" not in st.session_state:
    stored = load_readings()
    if not stored.empty:
        st.session_state["active_data"] = stored
    else:
        initial = generate_sample_data(days=30, seed=42)
        save_readings(initial)
        st.session_state["active_data"] = initial

df = st.session_state["active_data"].copy()
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

if len(df) < 48:
    st.warning("For useful daily forecasting, upload or generate at least 48 hourly readings.")

with st.spinner("Running forecasting and anomaly detection..."):
    result = analyze_energy(df)
    save_analysis(result["summary"], result["forecast"], result["anomalies"])

latest_day = df["timestamp"].max().date()
today_df = df[df["timestamp"].dt.date == latest_day]
today_kwh = float(today_df["kwh"].sum()) if not today_df.empty else float(df.tail(24)["kwh"].sum())
forecast_kwh = float(result["forecast"]["predicted_kwh"].sum()) if not result["forecast"].empty else 0.0
anomaly_count = int(result["anomalies"]["is_anomaly"].sum()) if not result["anomalies"].empty else 0
avg_hourly = float(df["kwh"].mean()) if len(df) else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest recorded day", f"{today_kwh:.2f} kWh")
m2.metric("Next-day forecast", f"{forecast_kwh:.2f} kWh")
m3.metric("Flagged readings", f"{anomaly_count}")
m4.metric("Average hourly use", f"{avg_hourly:.3f} kWh")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "AI Forecast", "Anomaly Detection", "Recommendations & History"])

with tab1:
    st.subheader("Energy consumption history")
    daily = df.set_index("timestamp")["kwh"].resample("D").sum().reset_index(name="daily_kwh")
    fig = px.line(daily, x="timestamp", y="daily_kwh", markers=True,
                  labels={"timestamp": "Date", "daily_kwh": "Daily consumption (kWh)"},
                  title="Daily electricity consumption")
    st.plotly_chart(fig, use_container_width=True)
    hourly = df.copy()
    hourly["hour"] = hourly["timestamp"].dt.hour
    profile = hourly.groupby("hour", as_index=False)["kwh"].mean()
    fig2 = px.bar(profile, x="hour", y="kwh",
                  labels={"hour": "Hour of day", "kwh": "Average hourly use (kWh)"},
                  title="Typical hourly usage profile")
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("View cleaned readings"):
        st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("Next 24-hour consumption forecast")
    forecast = result["forecast"]
    if forecast.empty:
        st.info("Not enough data to generate a forecast.")
    else:
        ffig = px.line(forecast, x="timestamp", y="predicted_kwh", markers=True,
                       labels={"timestamp": "Forecast time", "predicted_kwh": "Predicted use (kWh)"},
                       title="Forecast based on recent hourly patterns")
        st.plotly_chart(ffig, use_container_width=True)
        metrics = result["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Holdout MAE", "N/A" if metrics["mae"] is None else f"{metrics['mae']:.3f} kWh")
        c2.metric("Holdout RMSE", "N/A" if metrics["rmse"] is None else f"{metrics['rmse']:.3f} kWh")
        c3.metric("Model", metrics["model"])
        st.caption("Evaluation uses a chronological holdout. Lower MAE/RMSE generally indicates smaller forecast errors. Forecasts are estimates, not guarantees.")
        st.dataframe(forecast, use_container_width=True)

with tab3:
    st.subheader("Unusual consumption readings")
    anomalies = result["anomalies"]
    if anomalies.empty or anomaly_count == 0:
        st.success("No readings were flagged by the current anomaly model.")
    else:
        st.warning(f"{anomaly_count} readings were flagged for review. An anomaly is not proof of energy waste.")
        show = anomalies[anomalies["is_anomaly"]].copy()
        fig = px.scatter(anomalies, x="timestamp", y="kwh", color="is_anomaly",
                         color_discrete_map={True: "#d62728", False: "#1f77b4"},
                         labels={"timestamp": "Time", "kwh": "Hourly use (kWh)", "is_anomaly": "Flagged"},
                         title="Isolation Forest anomaly flags")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(show, use_container_width=True)

with tab4:
    st.subheader("Rule-based energy-saving suggestions")
    recommendations = build_recommendations(df, result["anomalies"], result["forecast"])
    for item in recommendations:
        with st.container(border=True):
            st.markdown(f"**{item['title']}**")
            st.write(item["detail"])
            st.caption(f"Priority: {item['priority']} · Evidence: {item['evidence']}")
    st.divider()
    st.subheader("Saved analysis history")
    history = load_analyses()
    if history.empty:
        st.info("Analysis history will appear here after analysis is saved.")
    else:
        st.dataframe(history, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download readings CSV", data=csv, file_name="ecopulse_readings.csv", mime="text/csv")
    if not result["forecast"].empty:
        fcsv = result["forecast"].to_csv(index=False).encode("utf-8")
        st.download_button("Download forecast CSV", data=fcsv, file_name="ecopulse_forecast.csv", mime="text/csv")

st.divider()
st.caption("EcoPulse AI research prototype • Synthetic readings are for demonstration only • Cloud-hosted SQLite may reset when the hosting instance restarts • No automatic control of appliances")
