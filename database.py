import sqlite3
import pandas as pd
from typing import Optional
from config import config 

def load_history() -> pd.DataFrame:
    """
    Load prediction history from database.
    
    Returns:
        DataFrame with historical data
    """
    try:
        conn = sqlite3.connect(config.DATA_BASE_PATH)
        query = "SELECT * FROM predictions ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except Exception as e:
        print(f"Error loading history: {e}")
        return pd.DataFrame()

def save_prediction(city:str, timestamp: str, prediction: float, temp: float, humidity: float, is_simulated=0):
    """
    Save a prediction to the database.
    
    Args:
        city: City name
        timestamp: Prediction timestamp
        prediction: Predicted value
        temp: Temperature
        humidity: Humidity
        is_simulated: Flag indicating if the prediction is simulated
    """
    try:
        conn = sqlite3.connect(config.DATA_BASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                city TEXT,
                temp REAL,
                humidity REAL,
                prediction REAL,
                is_simulated INTEGER
            )
        ''')
        
        cursor.execute('''
            INSERT OR REPLACE INTO predictions (timestamp, city, prediction, temp, humidity, is_simulated)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, city, prediction, temp, humidity, int(is_simulated)))
        df = pd.DataFrame([{
            'city': city,
            'temp': temp,
            'humidity': humidity,
            'prediction': prediction,
            'is_simulated': int(is_simulated)
        }])
        # Append the dataframe to the SQL table
        df.to_sql('predictions', conn, if_exists='append', index=False)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving prediction: {e}")