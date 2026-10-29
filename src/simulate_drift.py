import json
from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from src.monitor import (
    BASELINE_DATA_PATH,
    FEATURES,
    INFERENCE_BUFFER_PATH,
    load_clean_baseline,
    run_drift_evaluation,
)

RANDOM_STATE = 42
DRIFT_MULTIPLIER = 1.35
BENCHMARK_REPORT_PATH = Path("reports/recovery_benchmark.json")


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def generate_shifted_regime_labels(
    features_df: pd.DataFrame, base_labels: np.ndarray, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shifted_y = base_labels.copy()

    high_churn_mask = (features_df["MonthlyCharges"] > 95.0) & (features_df["tenure"] < 24)
    loyal_mask = (features_df["MonthlyCharges"] <= 85.0) & (features_df["tenure"] >= 24)

    shifted_y[high_churn_mask] = 1
    shifted_y[loyal_mask] = 0

    flip_indices = rng.random(len(shifted_y)) < 0.08
    shifted_y[flip_indices] = 1 - shifted_y[flip_indices]
    return shifted_y


def run_stress_test_and_recovery() -> dict:
    df = load_clean_baseline(BASELINE_DATA_PATH)
    df["Churn_Binary"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df[FEATURES].copy()
    y = df["Churn_Binary"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    baseline_clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    )
    baseline_clf.fit(X_train, y_train)

    preds_clean = baseline_clf.predict(X_test)
    probs_clean = baseline_clf.predict_proba(X_test)[:, 1]
    stage_a_metrics = evaluate_classifier(y_test, preds_clean, probs_clean)

    X_train_drifted = X_train.copy()
    X_test_drifted = X_test.copy()
    for part in (X_train_drifted, X_test_drifted):
        part["MonthlyCharges"] = np.round(part["MonthlyCharges"] * DRIFT_MULTIPLIER, 2)
        part["TotalCharges"] = np.round(part["TotalCharges"] * DRIFT_MULTIPLIER, 2)

    y_train_drifted = generate_shifted_regime_labels(X_train_drifted, y_train, seed=RANDOM_STATE)
    y_test_drifted = generate_shifted_regime_labels(X_test_drifted, y_test, seed=RANDOM_STATE + 1)

    INFERENCE_BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
    buffer_df = X_test_drifted.copy()
    buffer_df["churn_prediction"] = baseline_clf.predict(X_test_drifted)
    buffer_df["churn_probability"] = np.round(
        baseline_clf.predict_proba(X_test_drifted)[:, 1], 4
    )
    buffer_df.to_csv(INFERENCE_BUFFER_PATH, index=False)

    preds_degraded = baseline_clf.predict(X_test_drifted)
    probs_degraded = baseline_clf.predict_proba(X_test_drifted)[:, 1]
    stage_b_metrics = evaluate_classifier(y_test_drifted, preds_degraded, probs_degraded)

    drift_audit = run_drift_evaluation(trigger_retrain=False)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Customer_Churn_Baseline")

    with mlflow.start_run(run_name="rf_retrained_post_drift"):
        retrained_clf = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        )
        retrained_clf.fit(X_train_drifted, y_train_drifted)

        preds_recovered = retrained_clf.predict(X_test_drifted)
        probs_recovered = retrained_clf.predict_proba(X_test_drifted)[:, 1]
        stage_c_metrics = evaluate_classifier(y_test_drifted, preds_recovered, probs_recovered)

        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": RANDOM_STATE,
            "retrained_on_drift": "true",
            "drift_factor": "+35%_MonthlyCharges",
        })
        mlflow.log_metrics(stage_c_metrics)
        mlflow.sklearn.log_model(retrained_clf, artifact_path="model")

    print("\n" + "=" * 86)
    print(" CHAPTER 6 BENCHMARK: SILENT DEGRADATION (+35% DRIFT) & AUTONOMOUS RECOVERY DELTA")
    print("=" * 86)
    header = (
        f"{'Metric':<12} | {'(A) Baseline Clean':<20} | "
        f"{'(B) Baseline Drifted':<20} | {'(C) Retrained':<14} | {'Recovery (C - B)'}"
    )
    print(header)
    print("-" * 86)

    deltas = {}
    for metric_name in ["accuracy", "roc_auc", "precision", "recall", "f1_score"]:
        val_a = stage_a_metrics[metric_name]
        val_b = stage_b_metrics[metric_name]
        val_c = stage_c_metrics[metric_name]
        delta_val = round(val_c - val_b, 4)
        deltas[metric_name] = delta_val
        print(
            f"{metric_name:<12} | {val_a:<20.4f} | "
            f"{val_b:<20.4f} | {val_c:<14.4f} | {delta_val:+.4f}"
        )
    print("=" * 86)

    BENCHMARK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_summary = {
        "stage_a_baseline_clean": stage_a_metrics,
        "stage_b_baseline_drifted": stage_b_metrics,
        "stage_c_retrained_drifted": stage_c_metrics,
        "recovery_delta_c_minus_b": deltas,
        "drift_audit": drift_audit,
    }
    BENCHMARK_REPORT_PATH.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    print(f"[INFO] Saved recovery benchmark report to: {BENCHMARK_REPORT_PATH}\n")
    return full_summary


if __name__ == "__main__":
    run_stress_test_and_recovery()