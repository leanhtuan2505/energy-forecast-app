import pandas as pd
from database import get_recent_actuals_and_preds # Hypothetical helper
import subprocess # To run train_lstm.py

# 1. CONFIGURATION
MAE_THRESHOLD = 500.0  # If error > 500 MW, we retrain
WINDOW_SIZE = 24       # Check the last 24 hours of performance

def check_model_health():
    print("Evaluating model health...")
    
    # 2. FETCH RECENT DATA
    # You need a function in database.py that joins 'actual' vs 'predicted'
    df, metadata = get_recent_actuals_and_preds(limit=WINDOW_SIZE)
    
    if df.empty or len(df) < WINDOW_SIZE:
        print("Not enough data to evaluate health yet.")
        return False

    # 3. CALCULATE METRICS
    df['error'] = abs(df['actual'] - df['predicted'])
    current_mae = df['error'].mean()
    
    print(f"Current MAE: {current_mae:.2f} MW (Threshold: {MAE_THRESHOLD})")

    # 4. THE DECISION
    if current_mae > MAE_THRESHOLD:
        print("🚨 Performance degraded. Triggering retraining...")
        # Execute the training script as a subprocess
        subprocess.run(["python", "train_lstm.py"], check=True)
        return True
    else:
        print("✅ Model health is optimal. No retraining required.")
        return False

if __name__ == "__main__":
    check_model_health()