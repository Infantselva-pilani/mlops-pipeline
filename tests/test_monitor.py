import numpy as np
import pandas as pd
from src.monitor import calculate_psi, run_drift_evaluation

def test_psi_near_zero_for_identical_distributions():
    np.random.seed(42)
    baseline = np.random.normal(loc=50, scale=10, size=1000)
    target = np.random.normal(loc=50, scale=10, size=1000)
    
    psi_val = calculate_psi(baseline, target)
    assert psi_val < 0.05

def test_monitor_reports_stable_when_no_drift(tmp_path, monkeypatch):
    baseline_df = pd.DataFrame({
        "tenure": np.random.randint(1, 72, size=100),
        "MonthlyCharges": np.random.uniform(20, 100, size=100),
        "TotalCharges": np.random.uniform(100, 5000, size=100)
    })
    
    buffer_df = baseline_df.copy()
    
    base_file = tmp_path / "churn.csv"
    buff_file = tmp_path / "inference_buffer.csv"
    baseline_df.to_csv(base_file, index=False)
    buffer_df.to_csv(buff_file, index=False)
    
    monkeypatch.setattr("os.path.join", lambda *args: str(tmp_path / args[-1]) if args[-1] in ["churn.csv", "inference_buffer.csv"] else "/".join(args))
    
    # Run evaluation
    has_drift = run_drift_evaluation()
    assert has_drift is False

def test_monitor_detects_positive_35_percent_charge_drift(tmp_path, monkeypatch):
    np.random.seed(42)
    baseline_df = pd.DataFrame({
        "tenure": np.random.randint(1, 72, size=500),
        "MonthlyCharges": np.random.uniform(20, 100, size=500),
        "TotalCharges": np.random.uniform(100, 5000, size=500)
    })
    
    buffer_df = baseline_df.copy()
    buffer_df["MonthlyCharges"] = buffer_df["MonthlyCharges"] * 1.35
    
    base_file = tmp_path / "churn.csv"
    buff_file = tmp_path / "inference_buffer.csv"
    baseline_df.to_csv(base_file, index=False)
    buffer_df.to_csv(buff_file, index=False)
    
    monkeypatch.setattr("os.path.join", lambda *args: str(tmp_path / args[-1]) if args[-1] in ["churn.csv", "inference_buffer.csv"] else "/".join(args))
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)
    
    has_drift = run_drift_evaluation()
    assert has_drift is True