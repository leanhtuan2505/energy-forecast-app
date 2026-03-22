import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from database import fetch_all_training_data # Your existing data helper
from model import EnergyLSTM # Shared architecture

def train_model():
    print("--- Starting LSTM Training Sequence ---")
    
    # 1. Load and Preprocess Data
    df = fetch_all_training_data()
    if df.empty:
        print("Error: No training data found.")
        return

    scaler = MinMaxScaler()
    # Use float32 to save memory on GitHub Runners
    scaled_data = scaler.fit_transform(df[['consumption']].values.astype('float32'))
    
    # Save scaler for inference parity (Critical for accuracy)
    joblib.dump(scaler, 'scaler.pkl')

    # 2. Create Sequences (Windowing)
    def create_sequences(data, seq_length=24):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X, y = create_sequences(scaled_data)
    X_train = torch.from_numpy(X)
    y_train = torch.from_numpy(y)

    # 3. Model Initialization
    model = EnergyLSTM(input_size=1, hidden_layer_size=100, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop (Optimized for GH Actions CPU)
    epochs = 10 
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # 5. Save the Artifact (Corrected Attribute Name)
    torch.save(model.state_dict(), 'energy_lstm.pth')
    print("Training complete. Artifact 'energy_lstm.pth' saved successfully.")

if __name__ == "__main__":
    train_model()