import pandas as pd
from xgboost import XGBRegressor
import joblib
import holidays

us_holidays = holidays.US()

def train_new_model():
    # 1. Load Data
    df = pd.read_csv('data/PJME_hourly.csv', parse_dates=['Datetime'], index_col='Datetime')

    # 2. Feature Engineering
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    # For simplicity, we assume you have merged weather data here
    # If you haven't, the model will just use time features
    df['temp'] = 20.0  # Placeholder: Replace with your actual weather column
    df['humidity'] = 50.0 # Placeholder: Replace with your actual humidity column
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_holiday'] = df['Datetime'].apply(lambda x: 1 if x in us_holidays else 0)

    features = ['hour', 'dayofweek', 'month', 'temp', 'humidity', 'is_weekend', 'is_holiday']
    X = df[features]
    y = df['PJME_MW']

    # 3. Train
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
    model.fit(X, y)

    # 4. Save
    joblib.dump(model, 'energy_weather_model.pkl')
    print("Model saved as energy_weather_model.pkl")

if __name__ == "__main__":
    train_new_model()