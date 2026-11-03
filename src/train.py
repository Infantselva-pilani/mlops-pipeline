import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

def run_training():
    # Set tracking URI and experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Customer_Churn_Baseline")

    # Load dataset
    data_path = os.path.join("data", "churn.csv")
    df = pd.read_csv(data_path)

    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X = df[features]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="rf_baseline"):
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log model with skops_trusted_types fix and non-deprecated argument
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        print("Training completed and model logged successfully.")

if __name__ == "__main__":
    run_training()