import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

def display_current_prediction(prediction: float):
    """Display current energy demand prediction."""
    st.metric("Predicted Load", f"{int(prediction)} MW")

def display_anomaly_alerts(anomalies_count: int, max_peak: Optional[pd.Series]):
    """Display anomaly alerts if any exist."""
    if anomalies_count > 0:
        st.error(f"🚨 ALERT: {anomalies_count} high-demand anomalies detected in the next 5 days!")
        if max_peak is not None:
            from utils import format_datetime_for_display
            st.warning(f"Critical Peak Expected: **{int(max_peak['prediction_mw'])} MW** on {format_datetime_for_display(max_peak['datetime'])}")
    else:
        st.success("✅ Grid Stability: No major demand spikes forecasted.")

def display_forecast_chart(df_weather: pd.DataFrame):
    """Display the forecast line chart with threshold."""
    st.line_chart(df_weather.set_index('datetime')[['prediction_mw', 'threshold']])

def display_summary_table(summary_df: pd.DataFrame):
    """Display the 5-day summary table."""
    st.subheader("📅 5-Day Summary Table")
    st.dataframe(summary_df.style.format("{:.0f}"))

def display_history_chart(df_history: pd.DataFrame):
    """Display historical predictions chart."""
    required_cols = ['timestamp', 'prediction', 'temp']
    if df_history.empty:
        st.info("The database is currently empty.")
        return
    if not all(col in df_history.columns for col in required_cols):
        st.error("Missing required columns in history data.")
        return
    
    # Clean data: drop rows with invalid timestamps and sort
    df_history = df_history.dropna(subset=['timestamp']).copy()
    df_history = df_history.sort_values('timestamp')
    
    # Ensure plotted columns are numeric
    df_history['prediction'] = pd.to_numeric(df_history['prediction'], errors='coerce')
    df_history['temp'] = pd.to_numeric(df_history['temp'], errors='coerce')
    
    # Plot prediction and temp; add humidity if available
    cols_to_plot = ['prediction', 'temp']
    if 'humidity' in df_history.columns:
        df_history['humidity'] = pd.to_numeric(df_history['humidity'], errors='coerce')
        cols_to_plot.append('humidity')
    
    # Drop rows with NaN in plotted columns
    df_history = df_history.dropna(subset=cols_to_plot)
    
    if df_history.empty:
        st.info("No valid data to display after cleaning.")
        return
    
    st.line_chart(df_history.set_index('timestamp')[cols_to_plot])