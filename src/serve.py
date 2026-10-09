from contextlib import asynccontextmanager
from typing import Dict, Any
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import joblib

ml_models: Dict[str, Any] = {}

class ChurnRequest(BaseModel):
    tenure: int = Field(..., ge=0, le=120, description="Months customer has stayed")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly billing amount")
    TotalCharges: float = Field(..., ge=0.0, description="Total billed amount to date")

def _load_artifact_safely(model_dir: Path):
    """Safely loads model artifacts across skops, joblib, and MLflow."""
    # 1. If serialized with skops, explicitly trust scikit-learn tree nodes
    skops_path = model_dir / "model.skops"
    if skops_path.exists():
        try:
            import skops.io
            untrusted = skops.io.get_untrusted_types(file=str(skops_path))
            trusted_types = list(set(untrusted + ["sklearn.tree._tree.Tree"]))
            return skops.io.load(str(skops_path), trusted=trusted_types)
        except Exception as e:
            print(f"Direct skops loading failed: {e}. Trying alternatives...")

    # 2. If serialized with standard joblib/pickle
    pkl_path = model_dir / "model.pkl"
    if pkl_path.exists():
        return joblib.load(str(pkl_path))

    # 3. Fallback: try MLflow loader with trusted types passed
    try:
        return mlflow.sklearn.load_model(str(model_dir), skops_trusted_types=["sklearn.tree._tree.Tree"])
    except TypeError:
        return mlflow.sklearn.load_model(str(model_dir))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Connect to SQLite tracking store
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # 2. Locate the latest run ID from MLflow
    runs = mlflow.search_runs(
        experiment_names=["Customer_Churn_Baseline"],
        order_by=["start_time DESC"],
        max_results=1
    )
    if runs.empty:
        raise RuntimeError("No trained model found in MLflow database.")

    latest_run_id = runs.iloc[0]["run_id"]
    print(f"Targeting model from run: {latest_run_id}")

    # 3. Resilient model loading
    try:
        model_uri = f"runs:/{latest_run_id}/model"
        ml_models["model"] = mlflow.sklearn.load_model(
            model_uri,
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
    except Exception as e:
        print(f"Standard URI resolution failed ({e}). Resolving via local mlruns directory...")
        candidates = list(Path("mlruns").rglob("MLmodel"))
        if not candidates:
            raise RuntimeError("No MLmodel artifacts found inside the mlruns directory.")
        
        matching = [c.parent for c in candidates if latest_run_id in str(c)]
        model_dir = matching[0] if matching else max(candidates, key=lambda p: p.stat().st_mtime).parent
        
        print(f"Loading model artifact directly from: {model_dir}")
        ml_models["model"] = _load_artifact_safely(model_dir)

    print("Model loaded successfully into memory.")
    yield
    ml_models.clear()

app = FastAPI(
    title="Customer Churn Prediction Service",
    description="MLOps Real-Time Inference Microservice",
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
        raise HTTPException(status_code=503, detail="Model artifact unavailable")

    df = pd.DataFrame([payload.model_dump()])
    prob = float(ml_models["model"].predict_proba(df)[0][1])
    pred = int(prob >= 0.5)

    return {
        "churn_prediction": pred,
        "churn_probability": round(prob, 4),
        "will_churn": bool(pred == 1)
    }