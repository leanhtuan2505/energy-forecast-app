import sqlite3
import pandas as pd

DB_NAME = "energy_data.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Create the table if it doesn't exist
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
    conn.commit()
    conn.close()

def save_prediction(city, temp, humidity, prediction, is_simulated=0):
    conn = sqlite3.connect(DB_NAME)
    init_db()
    df = pd.DataFrame([{
        'city': city,
        'temp': temp,
        'humidity': humidity,
        'prediction': prediction,
        'is_simulated': int(is_simulated)
    }])
    # Append the dataframe to the SQL table
    df.to_sql('predictions', conn, if_exists='append', index=False)
    conn.close()

def load_history():
    conn = sqlite3.connect(DB_NAME)
    init_db()
    df = pd.read_sql('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 100', conn)
    conn.close()
    return df