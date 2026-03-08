import pandas as pd
from xgboost import XGBRegressor
import joblib
import holidays

from config import config


us_holidays = holidays.US()

def train_new_model():
    # 1. Load Data
    df = pd.read_csv('data/PJME_hourly.csv', parse_dates=['Datetime'])

    # 2. Feature Engineering
    df['hour'] = df['Datetime'].dt.hour
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['month'] = df['Datetime'].dt.month
    # For simplicity, we assume you have merged weather data here
    # If you haven't, the model will just use time features
    df['temp'] = 20.0  # Placeholder: Replace with your actual weather column
    df['humidity'] = 50.0 # Placeholder: Replace with your actual humidity column
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    # Convert the holiday list keys to a set of dates for faster and more reliable matching

    df['is_holiday'] = df['Datetime'].dt.date.apply(lambda x: 1 if x in us_holidays else 0)
    features = ['hour', 'dayofweek', 'month', 'temp', 'humidity', 'is_weekend', 'is_holiday']
    X = df[features]
    y = df['PJME_MW']

    # 3. Train
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
    model.fit(X, y)

    # 4. Save
    joblib.dump(model, config.MODEL_PATH)
    print(f"Model saved as {config.MODEL_PATH}")

if __name__ == "__main__":
    train_new_model()