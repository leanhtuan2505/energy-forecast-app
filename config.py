import os
import streamlit as st
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Logs to console; add FileHandler for files
)

class Config:
    @property
    def API_KEY(self):
        """Get OpenWeather API key from environment or Streamlit secrets."""
        key = os.getenv("OPENWEATHER_API_KEY") # or st.secrets.get("OPENWEATHER_API_KEY")
        if not key:
            raise ValueError("OPENWEATHER_API_KEY not found in environment or secrets")
        return key
    
    MODEL_PATH = "energy_weather_model.pkl"
    ALERT_THRESHOLD_MULTIPLIER = 1.5
    FORECAST_API_URL = "http://api.openweathermap.org/data/2.5/forecast"
    WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"
    UNITS = "metric"
    WEEKEND_START_DAY = 5  # Saturday
    DATA_BASE_PATH = "energy_demand_history.db"

    CITIES = {
        "Philadelphia": "Philadelphia,US",
        "New York": "New York,US", 
        "Chicago": "Chicago,US",
        "Washington DC": "Washington,US"
    }

# Create a global config instance
config = Config()