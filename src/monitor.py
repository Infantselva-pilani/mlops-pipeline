import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

BASELINE_DATA_PATH = Path("data/churn.csv")
INFERENCE_BUFFER_PATH = Path("data/inference_buffer.csv")
DRIFT_REPORT_PATH = Path("reports/drift_audit_report.json")
FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

KS_P_VALUE_THRESHOLD = 0.05
PSI_ALERT_THRESHOLD = 0.20


def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Calculates the Population Stability Index (PSI) between reference and live distributions."""
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_ratios = np.clip(ref_counts / max(len(reference), 1), 1e-4, 1.0)
    cur_ratios = np.clip(cur_counts / max(len(current), 1), 1e-4, 1.0)

    psi_val = np.sum((cur_ratios - ref_ratios) * np.log(cur_ratios / ref_ratios))
    return float(psi_val)


def load_clean_baseline(csv_path: Path = BASELINE_DATA_PATH) -> pd.DataFrame:
    """Loads and sanitizes the baseline Telco Customer Churn dataset."""
    df = pd.read_csv(csv_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=FEATURES)
    return df


def run_drift_evaluation(
    baseline_path: Path = BASELINE_DATA_PATH,
    buffer_path: Path = INFERENCE_BUFFER_PATH,
    trigger_retrain: bool = True,
) -> dict:
    """Executes Two-Sample KS-Test and PSI checks on buffered live inference traffic."""
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
    if not buffer_path.exists():
        raise FileNotFoundError(f"Inference buffer not found: {buffer_path}")

    ref_df = load_clean_baseline(baseline_path)
    live_df = pd.read_csv(buffer_path).dropna(subset=FEATURES)

    print("\n" + "=" * 76)
    print(" STATISTICAL COVARIATE SHIFT ENGINE (TWO-SAMPLE KS-TEST & PSI)")
    print("=" * 76)
    print(f"Baseline Reference Records : {len(ref_df)}")
    print(f"Buffered Live Payloads     : {len(live_df)}")
    print("-" * 76)
    print(f"{'Feature':<18} | {'D_KS Stat':<10} | {'KS p-value':<12} | {'PSI':<8} | {'Status'}")
    print("-" * 76)

    overall_drift_detected = False
    feature_results = {}

    for feature in FEATURES:
        ref_arr = ref_df[feature].to_numpy(dtype=float)
        live_arr = live_df[feature].to_numpy(dtype=float)

        ks_stat, p_value = ks_2samp(ref_arr, live_arr)
        psi_score = calculate_psi(ref_arr, live_arr)

        is_drifted = (p_value < KS_P_VALUE_THRESHOLD) or (psi_score > PSI_ALERT_THRESHOLD)
        if is_drifted:
            overall_drift_detected = True

        status_label = "DRIFT DETECTED" if is_drifted else "STABLE"
        print(f"{feature:<18} | {ks_stat:<10.4f} | {p_value:<12.4e} | {psi_score:<8.4f} | {status_label}")

        feature_results[feature] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": float(p_value),
            "psi": round(float(psi_score), 4),
            "drift_detected": bool(is_drifted),
        }

    print("=" * 76)

    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_payload = {
        "overall_drift_detected": overall_drift_detected,
        "ks_p_value_threshold": KS_P_VALUE_THRESHOLD,
        "psi_alert_threshold": PSI_ALERT_THRESHOLD,
        "baseline_samples": int(len(ref_df)),
        "buffered_samples": int(len(live_df)),
        "features": feature_results,
    }
    DRIFT_REPORT_PATH.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    print(f"[INFO] Saved drift audit report to: {DRIFT_REPORT_PATH}")

    if overall_drift_detected and trigger_retrain:
        print("[WEBHOOK ALERT] Statistical drift threshold breached (p < 0.05 or PSI > 0.20)!")
        print("[WEBHOOK ACTION] Executing automated continuous retraining (src/train.py)...")
        subprocess.run([sys.executable, "src/train.py"], check=True)
        print("[WEBHOOK SUCCESS] Retraining finished and logged to MLflow registry.")

    return audit_payload


if __name__ == "__main__":
    run_drift_evaluation(trigger_retrain=True)