import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib
from model import EnergyLSTM  # Ensure model.py is in the same folder

# 1. LOAD & PRE-PROCESS (Memory Optimized)
print("Loading data...")
# Only load the target column to save RAM
df = pd.read_csv('data/PJME_hourly.csv', usecols=['PJME_MW'])
data = df['PJME_MW'].values.astype(np.float32).reshape(-1, 1)

# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
joblib.dump(scaler, "scaler.pkl") # Save for Streamlit app use

# 2. WINDOWING (Sliding Window)
def create_sequences(data, window_size=24):
    num_samples = len(data) - window_size
    X = np.zeros((num_samples, window_size, 1), dtype=np.float32)
    y = np.zeros((num_samples, 1), dtype=np.float32)
    
    for i in range(num_samples):
        X[i] = data[i:i + window_size]
        y[i] = data[i + window_size]
    return torch.from_numpy(X), torch.from_numpy(y)

X_train, y_train = create_sequences(scaled_data)

# 3. BATCHING (The fix for your Memory Error)
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 4. MODEL INITIALIZATION
model = EnergyLSTM(input_size=1, hidden_size=64, num_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. TRAINING LOOP
print(f"Starting Training on {len(X_train)} samples...")
epochs = 20 # Start with 20 to test stability
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(loader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

# 6. EXPORT
torch.save(model.state_dict(), "energy_lstm_model.pth")
print("✅ Training complete. Assets 'energy_lstm_model.pth' and 'scaler.pkl' are ready.")