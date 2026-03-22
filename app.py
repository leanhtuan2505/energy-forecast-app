import streamlit as st
import torch
import pandas as pd
import numpy as np
import joblib
from model import EnergyLSTM
from database import get_recent_actuals_and_preds # Assuming this helper exists to pull from Supabase

# Page Configuration
st.set_page_config(page_title="Energy Forecast AI", layout="wide")

@st.cache_resource
def load_assets():
    """Load the model and scaler. Handles the case where they don't exist yet."""
    try:
        model = EnergyLSTM(input_size=1, hidden_layer_size=100, output_size=1)
        model.load_state_dict(torch.load('energy_lstm.pth', map_location=torch.device('cpu')))
        model.eval()
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

def main():
    st.title("⚡ Real-Time Energy Demand Forecast (LSTM)")
    st.markdown("---")

    model, scaler = load_assets()

    if model is None:
        st.error("⚠️ Model artifacts not found. Please run 'train_lstm.py' to generate the baseline.")
        return

    # Layout: Sidebar for Status, Main for Charts
    with st.sidebar:
        st.header("System Status")
        st.success("LSTM Model: Loaded")
        st.info("Source: GitHub Actions Pipeline")

    # 1. Fetch and Display Data
    data = get_recent_actuals_and_preds(limit=48) # Pull last 48 hours
    if not data.empty:
        st.subheader("Last 48 Hours: Actual vs. Predicted")
        st.line_chart(data[['actual_value', 'predicted_value']])

        # 2. Live Inference (The "Next Hour" Forecast)
        st.subheader("🔮 Next Hour Forecast")
        
        # Prepare the last sequence for LSTM
        last_sequence = data['actual_value'].values[-24:].reshape(-1, 1)
        scaled_seq = scaler.transform(last_sequence)
        input_tensor = torch.FloatTensor(scaled_seq).unsqueeze(0)

        with torch.no_grad():
            prediction_scaled = model(input_tensor)
            prediction = scaler.inverse_transform(prediction_scaled.numpy())

        col1, col2 = st.columns(2)
        col1.metric("Predicted Consumption", f"{prediction[0][0]:.2f} kWh")
        col2.metric("Confidence Level", "High" if len(data) > 24 else "Low")
    else:
        st.warning("No data found in Supabase. Check your 'fetch_data.py' logs.")

if __name__ == "__main__":
    main()