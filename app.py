import streamlit as st
from datetime import datetime
from config import config
from weather_api import get_live_weather
from prediction import prepare_prediction_features, predict_energy_demand
from utils import get_forecast_data, validate_city_input
from ui_components import (
    display_current_prediction, display_anomaly_alerts, 
    display_forecast_chart, display_summary_table, display_history_chart
)
from database import load_history

# Load model at startup
from prediction import load_model
model = load_model()

st.title("⚡ Energy Demand Predictor")

# 1. LIVE PREDICTION SECTION
if st.button("Get Forecast"):
    with st.spinner("Fetching current weather..."):
        temp, humidity = get_live_weather()
    
    if temp is not None:
        now = datetime.now()
        features_df = prepare_prediction_features(now, temp, humidity)
        prediction = predict_energy_demand(features_df)[0]
        display_current_prediction(prediction)
    else:
        st.error("Failed to fetch current weather data.")

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

st.divider()

# 3. PERFORMANCE & HISTORY SECTION
st.subheader("📊 Database History")
df_history = load_history()
display_history_chart(df_history)