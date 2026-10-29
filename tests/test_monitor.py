from pathlib import Path
import numpy as np
import pandas as pd
from src.monitor import calculate_psi, run_drift_evaluation


def test_psi_near_zero_for_identical_distributions():
    rng = np.random.default_rng(42)
    sample = rng.normal(loc=70.0, scale=15.0, size=1000)
    psi_val = calculate_psi(sample, sample)
    assert psi_val < 0.01


def test_monitor_reports_stable_when_no_drift(tmp_path: Path):
    base_csv = tmp_path / "base.csv"
    buf_csv = tmp_path / "buffer.csv"

    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "tenure": rng.integers(1, 72, size=500),
        "MonthlyCharges": rng.uniform(20.0, 100.0, size=500),
        "TotalCharges": rng.uniform(100.0, 5000.0, size=500),
        "Churn": ["No"] * 500,
    })
    df.to_csv(base_csv, index=False)
    df[["tenure", "MonthlyCharges", "TotalCharges"]].to_csv(buf_csv, index=False)

    report = run_drift_evaluation(
        baseline_path=base_csv, buffer_path=buf_csv, trigger_retrain=False
    )
    assert report["overall_drift_detected"] is False


def test_monitor_detects_positive_35_percent_charge_drift(tmp_path: Path):
    base_csv = tmp_path / "base.csv"
    buf_csv = tmp_path / "buffer.csv"

    rng = np.random.default_rng(42)
    base_df = pd.DataFrame({
        "tenure": rng.integers(1, 72, size=500),
        "MonthlyCharges": rng.uniform(20.0, 100.0, size=500),
        "TotalCharges": rng.uniform(100.0, 5000.0, size=500),
        "Churn": ["No"] * 500,
    })
    drifted_df = base_df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
    drifted_df["MonthlyCharges"] = drifted_df["MonthlyCharges"] * 1.35

    base_df.to_csv(base_csv, index=False)
    drifted_df.to_csv(buf_csv, index=False)

    report = run_drift_evaluation(
        baseline_path=base_csv, buffer_path=buf_csv, trigger_retrain=False
    )
    assert report["overall_drift_detected"] is True
    assert report["features"]["MonthlyCharges"]["drift_detected"] is True