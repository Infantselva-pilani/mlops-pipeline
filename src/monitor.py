import os
import sys
import json
import subprocess
import pandas as pd
from scipy.stats import ks_2samp
import numpy as np

def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_bins: int = 10) -> float:
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(baseline, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    baseline_counts, _ = np.histogram(baseline, bins=bins)
    target_counts, _ = np.histogram(target, bins=bins)
    
    baseline_pct = np.where(baseline_counts == 0, 0.0001, baseline_counts) / len(baseline)
    target_pct = np.where(target_counts == 0, 0.0001, target_counts) / len(target)
    
    psi_val = np.sum((target_pct - baseline_pct) * np.log(target_pct / baseline_pct))
    return float(psi_val)

def run_drift_evaluation():
    baseline_path = os.path.join("data", "churn.csv")
    buffer_path = os.path.join("data", "inference_buffer.csv")

    if not os.path.exists(buffer_path):
        print("No inference buffer found to audit.")
        return False

    base_df = pd.read_csv(baseline_path)
    buff_df = pd.read_csv(buffer_path)

    base_df['TotalCharges'] = pd.to_numeric(base_df['TotalCharges'], errors='coerce')
    base_df.dropna(inplace=True)

    features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    drift_detected = False
    audit_report = {}

    for f in features:
        ks_stat, p_val = ks_2samp(base_df[f], buff_df[f])
        psi_val = calculate_psi(base_df[f].values, buff_df[f].values)
        
        is_drifted = (p_val < 0.05) or (psi_val > 0.20)
        if is_drifted:
            drift_detected = True

        audit_report[f] = {
            "ks_stat": round(float(ks_stat), 4),
            "p_value": float(p_val),
            "psi": round(float(psi_val), 4),
            "drift_status": "DRIFT DETECTED" if is_drifted else "STABLE"
        }

    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "drift_audit_report.json"), "w") as f:
        json.dump(audit_report, f, indent=2)

    if drift_detected:
        print("Drift detected! Triggering continuous retraining pipeline...")
        subprocess.run([sys.executable, "src/train.py"], check=True)
        return True
    else:
        print("All features stable. No retraining required.")
        return False

# Alias for backwards compatibility
run_monitoring = run_drift_evaluation

if __name__ == "__main__":
    run_drift_evaluation()