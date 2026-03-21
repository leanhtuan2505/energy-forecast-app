import streamlit as st
import torch
import numpy as np
import pandas as pd
from model import EnergyLSTM # Our new shared 'Skeleton'
from database import get_recent_sequence # Our new 'Memory' fetcher
import joblib

# 1. LOAD ASSETS
# We still need the scaler from your training phase to normalize inputs
scaler = joblib.load("scaler.pkl") 

# Initialize the LSTM 'Skeleton'
model = EnergyLSTM(input_size=1, hidden_size=64, num_layers=2)

# Load the 'Muscles' (Weights)
try:
    model.load_state_dict(torch.load("energy_lstm_model.pth", map_location=torch.device('cpu')))
    model.eval()
    st.sidebar.success("🚀 LSTM Brain: Online")
except FileNotFoundError:
    st.sidebar.error("⚠️ Weights not found. Run train_lstm.py first.")

# 2. THE PREDICTION LOGIC
def predict_energy():
    # A. Get the last 24 hours of data from Supabase
    history = get_recent_sequence(limit=24)
    
    if len(history) < 24:
        st.warning(f"Need 24 hours of data. Currently have: {len(history)}")
        return None

    # B. Pre-process (Scale and Reshape for PyTorch)
    # Shape: [1, 24, 1] -> (Batch, Time-steps, Features)
    seq = np.array(history).reshape(-1, 1)
    scaled_seq = scaler.transform(seq)
    input_tensor = torch.FloatTensor(scaled_seq).view(1, 24, 1)

    # C. Inference
    with torch.no_grad():
        prediction_scaled = model(input_tensor)
    
    # D. Inverse Scale (Convert 0.1 back to 15,000 MW)
    prediction = scaler.inverse_transform(prediction_scaled.numpy())
    return prediction[0][0]

# 3. UI LAYOUT
st.title("⚡ AI Energy Forecaster (V2: Deep Learning)")

if st.button("Generate Forecast"):
    result = predict_energy()
    if result:
        st.metric("Predicted Load", f"{result:,.2f} MW")