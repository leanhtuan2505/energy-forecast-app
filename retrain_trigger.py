import os
import subprocess
from database import get_recent_actuals_and_preds # Resolved Import
from sklearn.metrics import mean_absolute_error

# ARCHITECTURAL CONSTANT: The threshold for 'Acceptable Error'
# If MAE (Mean Absolute Error) > 5.0 kWh, we trigger a retrain.
ERROR_THRESHOLD = 5.0 

def evaluate_and_trigger():
    print("--- Starting Self-Healing Diagnostic ---")
    
    # 1. Fetch data from Supabase (the function we just added)
    actuals, preds = get_recent_actuals_and_preds(limit=24)
    
    if not actuals or not preds or len(actuals) < 10:
        print("Insufficient data for evaluation. Skipping retrain.")
        return

    # 2. Calculate Performance Metric
    mae = mean_absolute_error(actuals, preds)
    print(f"Current Model MAE: {mae:.2f}")

    # 3. Decision Logic
    if mae > ERROR_THRESHOLD:
        print(f"ALERT: Error {mae:.2f} exceeds threshold {ERROR_THRESHOLD}.")
        print("Initiating LSTM Retraining Sequence...")
        
        try:
            # Trigger the training script as a subprocess
            result = subprocess.run(["python", "train_lstm.py"], check=True, capture_output=True, text=True)
            print("Retraining Successful:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"CRITICAL ERROR during retraining: {e.stderr}")
    else:
        print("Model performance is within acceptable parameters. No action required.")

if __name__ == "__main__":
    evaluate_and_trigger()