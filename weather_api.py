import requests
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple
from config import config
import logging

import logging

logger = logging.getLogger(__name__)

def get_live_weather(city: str = "Philadelphia") -> Tuple[Optional[float], Optional[float]]:
    """
    Fetch current weather data for a city.
    
    Args:
        city: City name for weather query
        
    Returns:
        Tuple of (temperature, humidity) or (None, None) on error
    """
    url = f"{config.WEATHER_API_URL}?q={city}&appid={config.API_KEY}&units={config.UNITS}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["main"]["temp"], data["main"]["humidity"]
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for {city}: {e}")
        return None, None
    except KeyError as e:
        logger.error(f"Unexpected API response format: {e}")
        return None, None

def get_7day_forecast(city_query: str) -> Optional[pd.DataFrame]:
    """
    Fetch 5-day weather forecast from OpenWeather API.
    
    Args:
        city_query: City query string (e.g., "Philadelphia,US")
        
    Returns:
        DataFrame with forecast data or None on error
    """
    url = f"{config.FORECAST_API_URL}?q={city_query}&appid={config.API_KEY}&units={config.UNITS}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        forecast_list = []
        for item in data['list']:
            forecast_list.append({
                "datetime": datetime.fromtimestamp(item['dt']),
                "temp": item['main']['temp'],
                "humidity": item['main']['humidity']
            })
        return pd.DataFrame(forecast_list)
    except requests.exceptions.RequestException as e:
        logger.error(f"Forecast API request failed: {e}")
        return None
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing forecast data: {e}")
        return None