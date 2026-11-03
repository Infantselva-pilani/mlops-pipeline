import os
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn

model = None

class ChurnRequest(BaseModel):
    tenure: int = Field(ge=0, le=120)
    MonthlyCharges: float = Field(ge=0.0)
    TotalCharges: float = Field(ge=0.0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Customer_Churn_Baseline")
    
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attribute.start_time DESC"],
            max_results=1
        )
        if runs:
            latest_run_id = runs[0].info.run_id
            model_uri = f"runs:/{latest_run_id}/model"
            # MLflow automatically reads the trusted types from the logged artifact
            model = mlflow.sklearn.load_model(model_uri=model_uri)
            print(f"Loaded model from run ID: {latest_run_id}")
    yield
    model = None

app = FastAPI(title="Telco Churn Prediction API", lifespan=lifespan)

def append_to_inference_buffer(record: dict, prediction: int, probability: float):
    buffer_path = os.path.join("data", "inference_buffer.csv")
    df = pd.DataFrame([{
        "tenure": record["tenure"],
        "MonthlyCharges": record["MonthlyCharges"],
        "TotalCharges": record["TotalCharges"],
        "prediction": prediction,
        "probability": probability
    }])
    df.to_csv(buffer_path, mode='a', header=not os.path.exists(buffer_path), index=False)

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(payload: ChurnRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    
    data = pd.DataFrame([payload.model_dump()])
    prob = float(model.predict_proba(data)[0][1])
    pred = int(prob >= 0.5)
    
    append_to_inference_buffer(payload.model_dump(), pred, prob)
    
    return {
        "churn_prediction": pred,
        "churn_probability": round(prob, 4)
    }