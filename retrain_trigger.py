import sqlite3
import pandas as pd
import joblib
from database import DB_NAME, init_db
from train_model import train_new_model

THRESHOLD = 2.5  # If Avg Error is > 2.5°C, we retrain

def check_model_health():
    conn = sqlite3.connect(DB_NAME)
    init_db()
    # Get the last 50 actual results vs predictions
    df = pd.read_sql("SELECT temp, prediction FROM predictions ORDER BY timestamp DESC LIMIT 50", conn)
    conn.close()

    if len(df) < 20:
        print("Not enough data to evaluate health yet.")
        return

    # Calculate Mean Absolute Error
    mae = (df['temp'] - df['prediction']).abs().mean()
    print(f"Current Model MAE: {mae:.2f}")

    if mae > THRESHOLD:
        print("⚠️ Accuracy too low! Triggering Retraining...")
        train_new_model() # This function runs your XGBoost fit() and saves the .pkl
    else:
        print("✅ Model health is stable. No retraining needed.")

if __name__ == "__main__":
    check_model_health()