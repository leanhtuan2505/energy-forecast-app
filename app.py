import streamlit as st
import pandas as pd
import requests
import joblib
from datetime import datetime

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('energy_weather_model.pkl')

model = load_model()

st.title("⚡ Energy Demand Predictor")


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
        
        # 1. Create the DataFrame
        input_data = pd.DataFrame({
            'hour': [now.hour],
            'dayofweek': [now.weekday()],
            'month': [now.month],
            'temp': [temp],
            'humidity': [humidity]
        })
        
        # 2. THE FIX: Force everything to be a number (float)
        input_data = input_data.astype(float) 
        
        # 3. Predict
        prediction = model.predict(input_data)[0]
        st.metric("Predicted Load", f"{int(prediction)} MW")