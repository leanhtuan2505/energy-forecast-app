import pandas as pd
from typing import Dict, Any
from datetime import datetime
import holidays

def create_summary_table(df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table grouped by date.
    
    Args:
        df_weather: DataFrame with weather and prediction data
        
    Returns:
        Summary DataFrame with daily statistics
    """
    df_weather['date'] = df_weather['datetime'].dt.date
    
    summary_df = df_weather.groupby('date').agg({
        'prediction_mw': ['mean', 'min', 'max'],
        'temp': 'mean'
    })
    
    summary_df.columns = ['Avg Demand (MW)', 'Min Demand (MW)', 'Max Demand (MW)', 'Avg Temp (°C)']
    return summary_df

def format_datetime_for_display(dt: datetime) -> str:
    """Format datetime for user display."""
    return dt.strftime('%A at %H:%M')

def validate_city_input(city: str) -> bool:
    """Validate that city input is in allowed cities."""
    from config import config
    return city in config.CITIES

def get_forecast_data(selected_city: str) -> Dict[str, Any]:
    """
    Generate complete forecast data for a city.
    
    Args:
        selected_city: City name from config.CITIES
        
    Returns:
        Dictionary with processed forecast data
    """
    from weather_api import get_7day_forecast
    from prediction import predict_energy_demand, detect_anomalies
    from config import config
    
    df_weather = get_7day_forecast(config.CITIES[selected_city])
    if df_weather is None:
        return {"error": "Failed to fetch forecast"}
    
    # Add time features
    df_weather['hour'] = df_weather['datetime'].dt.hour
    df_weather['dayofweek'] = df_weather['datetime'].dt.dayofweek
    df_weather['month'] = df_weather['datetime'].dt.month
    
    # Add weekend and holiday features
    us_holidays = holidays.US()
    df_weather['is_weekend'] = df_weather['dayofweek'].apply(lambda x: 1 if x >= config.WEEKEND_START_DAY else 0)
    df_weather['is_holiday'] = df_weather['datetime'].dt.date.apply(lambda x: 1 if x in us_holidays else 0)
    
    # Make predictions
    features = df_weather[['hour', 'dayofweek', 'month', 'temp', 'humidity', 'is_weekend', 'is_holiday']].astype(float)
    df_weather['prediction_mw'] = predict_energy_demand(features)

    # Detect anomalies
    anomalies, threshold = detect_anomalies(df_weather['prediction_mw'].tolist())
    df_weather['is_anomaly'] = anomalies
    df_weather['threshold'] = threshold
    
    # Create summary
    summary_df = create_summary_table(df_weather)
    
    return {
        "weather_df": df_weather,
        "summary_df": summary_df,
        "anomalies_count": sum(anomalies),
        "max_peak": df_weather.loc[df_weather['prediction_mw'].idxmax()] if anomalies else None
    }