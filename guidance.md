# Energy Consumption Predictor
Excellent choice. The **Energy Consumption Predictor** is a perfect starting point because it uses a "tabular" approach to time-series forecasting, which is much more intuitive than jumping straight into complex neural networks.

Here is your roadmap and a starter script to get the "real-time" engine running.

---

## The Workflow

1. **Historical Data:** We'll use the **PJM Hourly Energy** dataset. It contains years of power consumption in Megawatts (MW).
2. **External Features:** Energy use isn't just about time; it’s about **temperature**. People turn on AC when it’s hot. We’ll use the **OpenWeatherMap API** to get live weather.
3. **The App:** A **Streamlit** dashboard that polls the API and shows a "Current vs. Forecasted" chart.

---

## 1. Setup & API Key

You'll need a free API key from [OpenWeatherMap](https://openweathermap.org/api).

* Sign up and grab the **Current Weather Data** API key.
* Install the basics: `pip install streamlit pandas scikit-learn requests plotly`

## 2. The Project Structure

### Step A: The "Brain" (Training)

For a beginner, a **Random Forest Regressor** or **XGBoost** works best. You train it on features like:

* `hour` (0–23)
* `day_of_week` (0–6)
* `is_weekend` (Boolean)
* `temperature` (The most important external factor)

### Step B: The "Live" Script

This script simulates the real-time aspect by fetching the weather and making a prediction every time you refresh the page.

```python
import streamlit as st
import requests
import pandas as pd
import job pickle # Assuming you saved your trained model as 'model.pkl'

# 1. Configuration
API_KEY = "your_openweather_api_key"
CITY = "New York"
UNIT = "metric" # Use 'imperial' for Fahrenheit

st.title("⚡ Real-Time Energy Demand Forecast")

def get_live_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units={UNIT}"
    response = requests.get(url).json()
    return {
        "temp": response["main"]["temp"],
        "humidity": response["main"]["humidity"]
    }

# 2. Fetch Data
weather = get_live_weather(CITY)
now = pd.Timestamp.now()

# 3. Create Feature Row for the Model
input_data = pd.DataFrame([{
    "hour": now.hour,
    "day_of_week": now.dayofweek,
    "temp": weather["temp"],
    "humidity": weather["humidity"]
}])

# 4. Predict (Placeholder for your trained model)
# prediction = model.predict(input_data)
prediction = 1500 + (weather["temp"] * 10) # Simple logic for demo

# 5. Display
col1, col2 = st.columns(2)
col1.metric("Current Temp", f"{weather['temp']}°C")
col2.metric("Forecasted Demand", f"{prediction:.2f} MW")

st.write(f"Last updated: {now.strftime('%H:%M:%S')}")

```

---

## Key Challenges to Solve

* **Lag:** Real-time APIs have limits. In a real production environment, you would use a "Cache" so you don't ping the weather API every single second.
* **Data Alignment:** Your historical data (MW) and your weather data (Temp) must be from the same city/region to be accurate.


---

To build your model, we first need to prepare the historical data. The **PJM Hourly Energy** dataset is a gold standard for this. Since PJM spans multiple regions, we’ll use the `PJME` (PJM East) file as it’s the most commonly used for benchmarks.

Below is the "Phase 1" code to clean the data, engineer time-based features, and train a model you can actually use in your real-time app.

---

## 1. Data Preparation & Feature Engineering

In time-series forecasting, the model doesn't "know" what time it is. We have to break the timestamp into numerical features it can understand (e.g., hour 14 is usually higher demand than hour 03).

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset (Download 'PJME_hourly.csv' from Kaggle)
df = pd.read_csv('PJME_hourly.csv')
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

def create_features(df):
    """Creates time series features from datetime index."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

df = create_features(df)

# Split into Features (X) and Target (y)
# Note: In real life, you'd add weather data here as a column!
X = df[['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']]
y = df['PJME_MW']

# Train the model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

# Save the model to use in your Streamlit app later
joblib.dump(model, 'energy_model.pkl')
print("Model trained and saved!")

```

---

## 2. Refining the Real-Time Logic

Now that you have `energy_model.pkl`, your Streamlit app needs to match these features. If the model was trained on `hour`, the live app must provide the *current* `hour`.

### Pro-Tip: Adding Weather

To make this truly "real-time" and accurate, you should ideally train the model with a "Temperature" column. Since the PJM historical data doesn't include weather, you can:

1. **Merge data:** Find a historical weather CSV for New Jersey/Pennsylvania (the PJME area) and join it on the `Datetime` column.
2. **Predict with Weather:** In your live app, when you call the OpenWeatherMap API, you feed that live temperature into the `model.predict()` function.

---

## 3. Deployment Checklist

Before you move to the final dashboard, check these three things:

* **Timezone Alignment:** PJM data is usually in EST. Ensure your `pd.to_datetime` and your live API call use the same timezone to avoid a 5-hour forecast shift.
* **Normalization:** If you use a Neural Network later (like LSTM), you must scale your data ($0$ to $1$). For Random Forest, you don't need to worry about this.
* **Accuracy Metric:** Use **MAPE** (Mean Absolute Percentage Error). A good energy forecast usually has a MAPE under **5-7%**.

**Would you like me to show you how to merge a historical weather dataset with this energy data to improve your model's accuracy?** (This is the "secret sauce" that makes energy projects portfolio-ready).


---

To make your project truly professional, we need to merge the **Energy Consumption** data with **Historical Weather** data. A model that only knows the "hour" is just a clock; a model that knows the "temperature" is a forecaster.

### 1. The Data Merge Strategy

We will align both datasets by their **Timestamp**. This allows the model to learn the mathematical relationship:


$$High\ Temp + Summer\ Afternoon = AC\ Usage \uparrow$$

### 2. The Python Implementation

Assuming you have a weather CSV (like "NYC_Weather_2015_2023.csv") and your energy CSV:

```python
import pandas as pd

# 1. Load Datasets
df_energy = pd.read_csv('PJME_hourly.csv', parse_dates=['Datetime'], index_col='Datetime')
df_weather = pd.read_csv('weather_history.csv', parse_dates=['dt'], index_col='dt')

# 2. Resample Weather (Ensure it's hourly to match Energy)
# Some weather data is every 5 mins; we take the mean for the hour.
df_weather_hourly = df_weather[['temp', 'humidity']].resample('H').mean()

# 3. Inner Join (Merge only where timestamps exist in both)
df_combined = pd.merge(df_energy, df_weather_hourly, left_index=True, right_index=True, how='inner')

# 4. Check for Missing Values
df_combined = df_combined.ffill() # Forward fill any gaps in weather data

print(f"Combined Dataset Shape: {df_combined.shape}")
print(df_combined.head())

```

---

### 3. Training the "Weather-Aware" Model

Now, when we train the model, we include `temp` and `humidity`. This significantly lowers your **Error Rate**.

```python
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor # Upgrade to XGBoost for better performance

# Features: Time + Weather
features = ['hour', 'dayofweek', 'month', 'temp', 'humidity']
X = df_combined[features]
y = df_combined['PJME_MW']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)

# Save for the Streamlit App
import joblib
joblib.dump(model, 'energy_weather_model.pkl')

```

---

 

### Your Next Step

Your model is now ready for "Production." You have a file called `energy_weather_model.pkl`.

 
---
This is the "Grand Finale" of your project. We are going to weave the **OpenWeatherMap API**, your **trained XGBoost model**, and a **Streamlit** dashboard into one seamless real-time application.

### 1. The Real-Time Architecture

The app follows a simple loop: it fetches live weather, formats it to match your training data, and generates a prediction.

### 2. The Final "App.py" Code

Replace your previous script with this version. It is designed to work with the `energy_weather_model.pkl` we just created.

```python
import streamlit as st
import pandas as pd
import requests
import joblib
from datetime import datetime

# --- CONFIGURATION ---
API_KEY = "your_openweather_api_key_here"
CITY = "Philadelphia"  # Central to the PJM region
MODEL_PATH = "energy_weather_model.pkl"

# Load the brain
model = joblib.load(MODEL_PATH)

# --- FUNCTIONS ---
def get_live_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()
    return {
        "temp": response["main"]["temp"],
        "humidity": response["main"]["humidity"]
    }

def create_features(temp, humidity):
    now = datetime.now()
    # Ensure these names match the 'features' list from your training script!
    data = {
        'hour': [now.hour],
        'dayofweek': [now.weekday()],
        'month': [now.month],
        'temp': [temp],
        'humidity': [humidity]
    }
    return pd.DataFrame(data)

# --- USER INTERFACE ---
st.set_page_config(page_title="Live Energy Forecast", page_icon="⚡")
st.title("⚡ Real-Time Energy Demand Predictor")
st.write(f"Current Location: **{CITY}**")

if st.button('Update Forecast'):
    with st.spinner('Fetching live data...'):
        # 1. Get Live Data
        weather = get_live_weather()
        
        # 2. Process Features
        input_df = create_features(weather['temp'], weather['humidity'])
        
        # 3. Predict
        prediction = model.predict(input_df)[0]
        
        # 4. Display Results
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", f"{weather['temp']}°C")
        col2.metric("Humidity", f"{weather['humidity']}%")
        col3.metric("Predicted Load", f"{int(prediction)} MW", delta="Live Update")
        
        st.success(f"Forecast generated at {datetime.now().strftime('%H:%M:%S')}")
        
        # Optional: Add a simple bar chart
        st.bar_chart(input_df[['temp', 'humidity']])

```

 

---

### How to Run It

1. Make sure `energy_weather_model.pkl` is in the same folder as this script.
2. Open your terminal/command prompt.
3. Type: `streamlit run App.py`

 