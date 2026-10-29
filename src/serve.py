import csv
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Global model reference
model = None
BUFFER_CSV_PATH = Path("data/inference_buffer.csv")


class ChurnRequest(BaseModel):
    tenure: int = Field(..., ge=0, le=120, description="Customer tenure in months")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly recurring bill")
    TotalCharges: float = Field(..., ge=0.0, description="Cumulative charges")


def append_to_inference_buffer(
    tenure: int,
    monthly_charges: float,
    total_charges: float,
    churn_prediction: int,
    churn_probability: float,
) -> None:
    """Appends incoming live inference payloads to data/inference_buffer.csv for drift monitoring."""
    try:
        BUFFER_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_exists = BUFFER_CSV_PATH.exists()
        with open(BUFFER_CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "tenure",
                    "MonthlyCharges",
                    "TotalCharges",
                    "churn_prediction",
                    "churn_probability",
                ])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                int(tenure),
                round(float(monthly_charges), 2),
                round(float(total_charges), 2),
                int(churn_prediction),
                round(float(churn_probability), 4),
            ])
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Customer_Churn_Baseline")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time DESC"],
                max_results=1,
            )
            if runs:
                run_id = runs[0].info.run_id
                model_uri = f"runs:/{run_id}/model"
                model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        try:
            model = mlflow.sklearn.load_model("runs:/model")
        except Exception:
            pass
    yield
    model = None


app = FastAPI(title="Telecom Churn Prediction Service", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(payload: ChurnRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded into memory.")

    input_data = pd.DataFrame([{
        "tenure": payload.tenure,
        "MonthlyCharges": payload.MonthlyCharges,
        "TotalCharges": payload.TotalCharges,
    }])

    try:
        probabilities = model.predict_proba(input_data)[0]
        prediction = int(model.predict(input_data)[0])
        probability = float(probabilities[1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    append_to_inference_buffer(
        tenure=payload.tenure,
        monthly_charges=payload.MonthlyCharges,
        total_charges=payload.TotalCharges,
        churn_prediction=prediction,
        churn_probability=probability,
    )

    return {
        "churn_prediction": prediction,
        "churn_probability": probability,
        "will_churn": bool(prediction == 1),
    }