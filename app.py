import os

import streamlit as st
import pandas as pd
import requests
import joblib
import holidays
from datetime import datetime
from database import load_history

# 1. LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load('energy_weather_model.pkl')

model = load_model()

us_holidays = holidays.US()
now = datetime.now()

st.title("⚡ Energy Demand Predictor")

# 2. LIVE PREDICTION SECTION
CITIES = {
    "Philadelphia": "Philadelphia,US",
    "New York": "New York,US",
    "Chicago": "Chicago,US",
    "Washington DC": "Washington,US"
}

def get_7day_forecast(city_query):
    API_KEY = st.secrets["OPENWEATHER_API_KEY"]
    # Using the 5-day forecast API (free tier)
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city_query}&appid={API_KEY}&units=metric"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        forecast_list = []
        for item in data['list']:
            forecast_list.append({
                "datetime": datetime.fromtimestamp(item['dt']),
                "temp": item['main']['temp'],
                "humidity": item['main']['humidity']
            })
        return pd.DataFrame(forecast_list)
    else:
        st.error("Failed to fetch forecast.")
        return None
    
def get_live_weather(city="Philadelphia"):
    # Ensure your API key is active (can take 2 hours for new keys)
    API_KEY = st.secrets["OPENWEATHER_API_KEY"]
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # 1. Check if the request was successful (Status 200)
        if response.status_code == 200:
            return data["main"]["temp"], data["main"]["humidity"]
        else:
            # 2. If it failed, show the specific message from the API
            st.error(f"API Error: {data.get('message', 'Unknown Error')}")
            return None, None
            
    except Exception as e:
        st.error(f"Network Connection Error: {e}")
        return None, None
    
if st.button("Get Forecast"):
    temp, humidity = get_live_weather()
    
    # Check if we actually got data back
    if temp is not None:
        now = datetime.now()
        is_weekend = 1 if now.weekday() >= 5 else 0
        is_holiday = 1 if now in us_holidays else 0
        # 1. Create the DataFrame
        input_data = pd.DataFrame({
            'hour': [now.hour],
            'dayofweek': [now.weekday()],
            'month': [now.month],
            'temp': [temp],
            'humidity': [humidity],
            'is_weekend': [is_weekend],
            'is_holiday': [is_holiday]
        })
        
        # 2. THE FIX: Force everything to be a number (float)
        input_data = input_data.astype(float) 
        
        # 3. Predict
        prediction = model.predict(input_data)[0]
        st.metric("Predicted Load", f"{int(prediction)} MW")


# 1. City Selection
selected_city = st.selectbox("Select a City:", list(CITIES.keys()))

if st.button("Generate 5-Day Forecast"):
    df_weather = get_7day_forecast(CITIES[selected_city])
    
    if df_weather is not None:
        # 2. Prepare features for the model
        df_weather['hour'] = df_weather['datetime'].dt.hour
        df_weather['dayofweek'] = df_weather['datetime'].dt.dayofweek
        df_weather['month'] = df_weather['datetime'].dt.month
        
        # 3. Predict for all rows at once!
        # Features must match: ['hour', 'dayofweek', 'month', 'temp', 'humidity']
        X_live = df_weather[['hour', 'dayofweek', 'month', 'temp', 'humidity']].astype(float)
        df_weather['prediction_mw'] = model.predict(X_live)
        
        ###
        # 1. Calculate the Baseline (The "Normal" Level)
        avg_demand = df_weather['prediction_mw'].mean()
        std_demand = df_weather['prediction_mw'].std()

        # Define the "Alert Level" (e.g., Mean + 1.5 Standard Deviations)
        alert_threshold = avg_demand + (1.5 * std_demand)

        # 2. Find the Peaks
        peaks = df_weather[df_weather['prediction_mw'] > alert_threshold]

        # 3. UI Display
        if not peaks.empty:
            st.error(f"🚨 ALERT: {len(peaks)} high-demand anomalies detected in the next 5 days!")
            
            # Show the most extreme peak
            max_peak = peaks.loc[peaks['prediction_mw'].idxmax()]
            st.warning(f"Critical Peak Expected: **{int(max_peak['prediction_mw'])} MW** on {max_peak['datetime'].strftime('%A at %H:%M')}")
        else:
            st.success("✅ Grid Stability: No major demand spikes forecasted.")

        # 4. Visualize the Threshold on the Chart
        # We create a horizontal line for the threshold
        df_weather['threshold'] = alert_threshold
        st.line_chart(df_weather.set_index('datetime')[['prediction_mw', 'threshold']])


        ### ADD SUMMARY TABLE
        # 1. Group the 5-day forecast by Date
        df_weather['date'] = df_weather['datetime'].dt.date

        # 2. Calculate the Daily Statistics
        summary_df = df_weather.groupby('date').agg({
            'prediction_mw': ['mean', 'min', 'max'],
            'temp': 'mean'
        })

        # 3. Clean up the Column Names (Flatten the Multi-Index)
        summary_df.columns = ['Avg Demand (MW)', 'Min Demand (MW)', 'Max Demand (MW)', 'Avg Temp (°C)']

        # 4. Display the Summary Table
        st.subheader("📅 5-Day Summary Table")
        st.dataframe(summary_df.style.format("{:.0f}")) # Format to remove decimals

st.divider() # A visual line to separate current from past

# 3. PERFORMANCE & HISTORY SECTION
st.subheader("📊 Database History")
df_history = load_history()

if not df_history.empty:
    st.line_chart(df_history.set_index('timestamp')[['prediction', 'temp']])
else:
    st.info("The database is currently empty.")