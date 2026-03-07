# ⚡ Real-Time Energy Demand Forecast Engine

A real-time forecasting application that predicts grid-level electricity demand ($MW$) by integrating live meteorological data with a gradient-boosted regression model (**XGBoost**).

## 📌 Project Overview

The goal of this project is to solve the "Time + Environment" challenge in energy forecasting. Traditional models often rely solely on historical patterns; this engine pulls **live weather data** (temperature and humidity) via the **OpenWeatherMap API** to adjust its predictions dynamically based on current climate conditions.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Model:** XGBoost Regressor (Gradient Boosting)
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Real-time API:** OpenWeatherMap (REST API)
* **Deployment:** Streamlit (Web Dashboard)

## 📊 The Data Pipeline

1. **Historical Training:** Trained on the **PJM Interconnection** dataset (Eastern region) covering 10+ years of hourly load data.
2. **Feature Engineering:** Extracted temporal features (Hour, Day of Week, Month) and merged them with historical weather observations.
3. **Live Inference:** 
    * The app pings the weather API for the current coordinates of the grid.
    * Features are transformed into a matching schema for the model.
    * The model returns a point-in-time demand forecast in Megawatts ($MW$).



## 🚀 How to Run

1. **Clone the repo:**
    ```bash
    git clone https://github.com/your-username/energy-forecast-project.git

    ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt

    ```
3. **Set your API Key:**
    Replace the `API_KEY` variable in `app.py` with your OpenWeatherMap key.
4. **Launch the Dashboard:**
    ```bash
    python train_model.py
    python -m streamlit run app.py

    ```

 