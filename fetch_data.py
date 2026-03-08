import pandas as pd
import joblib
from datetime import datetime
import holidays
from database import save_prediction
from config import config
import logging

logger = logging.getLogger(__name__)
from weather_api import get_live_weather  # Reuse existing API function

# 1. Load Model
model = joblib.load(config.MODEL_PATH)

us_holidays = holidays.US()

def get_data_and_save():
    # Fetch weather using the reusable function
    temp, humidity = get_live_weather("Philadelphia")
    
    if temp is not None and humidity is not None:
        now = datetime.now()
        is_weekend = 1 if now.weekday() >= 5 else 0
        is_holiday = 1 if now in us_holidays else 0

        # 2. Predict
        input_data = pd.DataFrame([[
            now.hour, 
            now.weekday(), 
            now.month, 
            temp, 
            humidity, 
            is_weekend, 
            is_holiday]], 
            columns=['hour', 'dayofweek', 'month', 'temp', 'humidity', 'is_weekend', 'is_holiday'])
        prediction = model.predict(input_data)[0]        
           
        save_prediction(city="Philadelphia", timestamp=str(now), temp=temp, humidity=humidity, 
                prediction=prediction, is_simulated=0)
        logger.info("Data saved to SQLite database.")
    else:
        logger.error("Failed to fetch weather data.")

if __name__ == "__main__":
    get_data_and_save()