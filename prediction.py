import joblib
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import List, Tuple
from config import config

@st.cache_resource
def load_model():
    """Load the ML model with caching."""
    try:
        return joblib.load(config.MODEL_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {config.MODEL_PATH}")

def prepare_prediction_features(datetime_obj: datetime, temp: float, humidity: float, 
                               include_holiday: bool = True) -> pd.DataFrame:
    """
    Prepare feature DataFrame for model prediction.
    
    Args:
        datetime_obj: Datetime for prediction
        temp: Temperature value
        humidity: Humidity value
        include_holiday: Whether to include weekend/holiday features
        
    Returns:
        DataFrame with features for model
    """
    import holidays
    us_holidays = holidays.US()
    
    features = {
        'hour': datetime_obj.hour,
        'dayofweek': datetime_obj.weekday(),
        'month': datetime_obj.month,
        'temp': temp,
        'humidity': humidity
    }
    
    if include_holiday:
        features.update({
            'is_weekend': 1 if datetime_obj.weekday() >= config.WEEKEND_START_DAY else 0,
            'is_holiday': 1 if datetime_obj in us_holidays else 0
        })
    
    return pd.DataFrame([features]).astype(float)

def predict_energy_demand(features_df: pd.DataFrame) -> List[float]:
    """
    Make energy demand predictions using the ML model.
    
    Args:
        features_df: DataFrame with model features
        
    Returns:
        List of predictions
    """
    model = load_model()
    return model.predict(features_df).tolist()

def detect_anomalies(predictions: List[float], threshold_multiplier: float = None) -> Tuple[List[bool], float]:
    """
    Detect high-demand anomalies in predictions.
    
    Args:
        predictions: List of predicted values
        threshold_multiplier: Multiplier for standard deviation threshold
        
    Returns:
        Tuple of (anomaly flags, threshold value)
    """
    if threshold_multiplier is None:
        threshold_multiplier = config.ALERT_THRESHOLD_MULTIPLIER
        
    predictions_series = pd.Series(predictions)
    avg_demand = predictions_series.mean()
    std_demand = predictions_series.std()
    threshold = avg_demand + (threshold_multiplier * std_demand)
    
    anomalies = (predictions_series > threshold).tolist()
    return anomalies, threshold