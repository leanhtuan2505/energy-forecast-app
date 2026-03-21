import pandas as pd
from supabase import create_client, Client
from config import config

# These should be stored in GitHub Secrets and Streamlit Secrets!
url = config.SUPABASE_URL
key = config.SUPABASE_KEY
supabase: Client = create_client(url, key)

def save_prediction(city, temp, humidity, prediction, is_weekend=0, is_holiday=0, is_simulated=0, timestamp=None):
    data = {
        "city": city,
        "temp": temp,
        "humidity": humidity,
        "prediction": float(prediction), # Ensure it's not a numpy type
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "is_simulated": is_simulated
    }
    
    # If a custom timestamp is provided, use it; otherwise Supabase uses 'now'
    if timestamp:
        data["timestamp"] = timestamp

    return supabase.table("predictions").insert(data).execute()

def load_history():
    # Fetch the last 100 rows from the cloud
    response = supabase.table("predictions").select("*").order("timestamp", desc=True).limit(100).execute()
    return pd.DataFrame(response.data)


# ... (Keep your existing Supabase connection code)

def get_recent_sequence(limit=24):
    """Fetches the last N records to create a window for the LSTM."""
    try:
        response = supabase.table("predictions") \
            .select("temp") \
            .order("timestamp", desc=True) \
            .limit(limit) \
            .execute()
        
        # We need chronological order (Oldest -> Newest)
        # So we reverse the list returned by Supabase
        values = [row['temp'] for row in reversed(response.data)]
        return values
    except Exception as e:
        print(f"Error fetching sequence: {e}")
        return []