import pandas as pd
import requests
import joblib
from datetime import datetime
import os
import holidays
from database import init_db, save_prediction

# 1. Load Model
model = joblib.load('energy_weather_model.pkl')

us_holidays = holidays.US()
now = datetime.now()

def get_data_and_save():
    API_KEY = os.getenv("OPENWEATHER_API_KEY") # We use an Environment Variable
    city = "Philadelphia,US"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
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
           
        init_db()
        save_prediction(city="Philadelphia", temp=temp, humidity=humidity, 
                prediction=prediction, is_simulated=0)
        print("Data saved to SQLite database.")

if __name__ == "__main__":
    get_data_and_save()