import os
import pandas as pd
from src.monitor import run_drift_evaluation

def run_simulation():
    baseline_path = os.path.join("data", "churn.csv")
    buffer_path = os.path.join("data", "inference_buffer.csv")
    
    print("=" * 65)
    print(" INJECTING +35% MACROECONOMIC TARIFF INFLATION (DRIFT)")
    print("=" * 65)
    
    # Load the baseline
    df = pd.read_csv(baseline_path)
    
    # Create an unseen evaluation partition (20% = 1407 records)
    eval_df = df.sample(frac=0.20, random_state=42).copy()
    
    # Inject +35% drift into billing features while keeping tenure unperturbed
    eval_df['MonthlyCharges'] = eval_df['MonthlyCharges'] * 1.35
    eval_df['TotalCharges'] = pd.to_numeric(eval_df['TotalCharges'], errors='coerce') * 1.35
    
    # Save to the inference buffer
    eval_df.to_csv(buffer_path, index=False)
    
    print(f"[SIMULATOR] Sampled {len(eval_df)} records for the live inference buffer.")
    print("[SIMULATOR] Successfully scaled MonthlyCharges and TotalCharges by +35%.")
    print("[SIMULATOR] Triggering the statistical monitoring engine...\n")
    
    # Run the monitor to detect drift and trigger retraining
    run_drift_evaluation()

if __name__ == "__main__":
    run_simulation()