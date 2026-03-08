import streamlit as st
import pandas as pd
from datetime import datetime
import time
from config import config
from weather_api import get_live_weather
from prediction import prepare_prediction_features, predict_energy_demand
from utils import get_forecast_data, validate_city_input
from ui_components import (
    display_current_prediction, display_anomaly_alerts, 
    display_forecast_chart, display_summary_table, display_history_chart
)
from database import load_history
import logging

logger = logging.getLogger(__name__)

# Load model at startup
from prediction import load_model
model = load_model()

st.title("⚡ Energy Demand Predictor")

# 1. LIVE PREDICTION SECTION
if st.button("Get Forecast"):
    with st.spinner("Fetching current weather..."):
        try:
            temp, humidity = get_live_weather()
        
            if temp is not None:
                now = datetime.now()
                features_df = prepare_prediction_features(now, temp, humidity)
                prediction = predict_energy_demand(features_df)[0]
                display_current_prediction(prediction)
            else:
                st.error("Failed to fetch current weather data.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"UI error: {e}")

# 2. 5-DAY FORECAST SECTION
selected_city = st.selectbox("Select a City:", list(config.CITIES.keys()))

if st.button("Generate 5-Day Forecast"):
    if not validate_city_input(selected_city):
        st.error("Invalid city selection")
    else:
        with st.spinner("Generating forecast..."):
            forecast_data = get_forecast_data(selected_city)
        
        if "error" in forecast_data:
            st.error(forecast_data["error"])
        else:
            # Display alerts
            display_anomaly_alerts(forecast_data["anomalies_count"], forecast_data["max_peak"])
            
            # Display chart
            display_forecast_chart(forecast_data["weather_df"])
            
            # Display summary table
            display_summary_table(forecast_data["summary_df"])


# --- LIVE MONITORING SECTION ---
st.divider()
st.header("📊 Real-Time Prediction History")

# We wrap the table in a 'fragment' so it can refresh independently
@st.fragment(run_every="60s")
def show_live_table():
    try:
        # 1. Fetch data using your new Supabase function
        df_history = load_history()
        
        if not df_history.empty:
            # 2. Basic cleanup for the UI
            df_display = df_history[['timestamp', 'temp', 'prediction', 'city']].copy()
            df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%H:%M:%S')
            
            # 3. Display the interactive table
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            st.caption(f"Last updated: {time.strftime('%H:%M:%S')} (Auto-refreshes every 1m)")
        else:
            st.info("No data found in Supabase yet. Waiting for first GitHub Action run...")
            
    except Exception as e:
        st.error(f"Database Connection Error: {e}")

# Call the fragment
show_live_table()

st.divider()

# 3. PERFORMANCE & HISTORY SECTION
st.subheader("📊 Database History")
df_history = load_history()
display_history_chart(df_history)
