import pandas as pd
import requests
import joblib
from datetime import datetime
import os

# 1. Load Model
model = joblib.load('energy_weather_model.pkl')

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
        
        # 2. Predict
        input_data = pd.DataFrame([[now.hour, now.weekday(), now.month, temp, humidity]], 
                                 columns=['hour', 'dayofweek', 'month', 'temp', 'humidity'])
        prediction = model.predict(input_data)[0]
        
        # 3. Append to CSV
        new_row = pd.DataFrame([[now, temp, prediction]], columns=['timestamp', 'temp', 'prediction'])
        
        # If file exists, append; if not, create
        header = not os.path.exists('prediction_history.csv')
        new_row.to_csv('prediction_history.csv', mode='a', index=False, header=header)
        print(f"Recorded prediction: {prediction} MW at {now}")

if __name__ == "__main__":
    get_data_and_save()