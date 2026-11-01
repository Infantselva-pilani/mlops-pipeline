import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# MLflow tracking configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer_Churn_Baseline")


def run_training():

    with mlflow.start_run(run_name="rf_baseline"):

        # 1. Load dataset
        df = pd.read_csv("data/churn.csv")

        # 2. Data preprocessing
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

        df = df.dropna()

        # 3. Feature selection
        features = ["tenure", "MonthlyCharges", "TotalCharges"]

        X = df[features]
        y = df["Churn"]

        # 4. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # 5. Model configuration
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        }

        # 6. Model training
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # 7. Model predictions
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        # 8. Model evaluation
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(
                y_test, preds, zero_division=0
            ),
            "recall": recall_score(
                y_test, preds, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, preds, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, probs)
        }

        # 9. Log model parameters and evaluation metrics
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # 10. Generate model signature
        predictions = model.predict(X_train)
        signature = infer_signature(X_train, predictions)

        # 11. Log model to MLflow using secure skops serialization
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # 12. Display evaluation results
        print("\n--- Baseline Run Metrics ---")

        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        print("\nModel successfully trained and logged to MLflow.")


if __name__ == "__main__":
    run_training()
