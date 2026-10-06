from contextlib import asynccontextmanager
from typing import Dict, Any
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import mlflow.sklearn

# In-memory storage for the deserialized model
ml_models: Dict[str, Any] = {}

class ChurnRequest(BaseModel):
    tenure: int = Field(..., ge=0, le=120, description="Subscription duration in months")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly recurring bill")
    TotalCharges: float = Field(..., ge=0.0, description="Cumulative lifetime spend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Connect to MLflow SQLite database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # 2. Query the latest successful baseline run
    runs = mlflow.search_runs(
        experiment_names=["Customer_Churn_Baseline"],
        order_by=["start_time DESC"],
        max_results=1
    )
    if runs.empty:
        raise RuntimeError("No trained model found in MLflow. Run src/train.py first.")

    latest_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{latest_run_id}/model"
    
    # 3. Load the model into memory
    print(f"Loading model from run: {latest_run_id}")
    ml_models["model"] = mlflow.sklearn.load_model(model_uri)
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time inference endpoint for telecom customer churn scoring",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    if "model" in ml_models:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.post("/predict")
def predict(payload: ChurnRequest):
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model is not loaded in memory")

    # Format the incoming request as a single-row DataFrame
    input_df = pd.DataFrame([payload.model_dump()])
    
    # Compute churn probability
    churn_prob = float(ml_models["model"].predict_proba(input_df)[0][1])
    is_churn = int(churn_prob >= 0.5)

    return {
        "churn_prediction": is_churn,
        "churn_probability": round(churn_prob, 4),
        "will_churn": bool(is_churn == 1)
    }