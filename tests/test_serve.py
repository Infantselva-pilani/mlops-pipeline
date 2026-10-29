import pytest
from fastapi.testclient import TestClient
from src.serve import app

@pytest.fixture(scope="module")
def client():
    """
    Spins up the FastAPI app within its lifespan context.
    Loads the MLflow model into memory once for all tests in this file.
    """
    with TestClient(app) as test_client:
        yield test_client

# ---------------------------------------------------------------------------
# Test 1: Service Health & Model Loaded Check
# ---------------------------------------------------------------------------
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

# ---------------------------------------------------------------------------
# Test 2: Standard Prediction Flow & Output Contract
# ---------------------------------------------------------------------------
def test_predict_valid_record(client):
    payload = {
        "tenure": 12,
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    # Check expected keys exist
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "will_churn" in data
    
    # Mathematical bounds assertions
    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["will_churn"], bool)

# ---------------------------------------------------------------------------
# Test 3: Input Validation Defense (Negative tenure)
# ---------------------------------------------------------------------------
def test_predict_rejects_negative_tenure(client):
    payload = {
        "tenure": -5,  # Invalid: Pydantic specifies ge=0
        "MonthlyCharges": 50.0,
        "TotalCharges": 100.0
    }
    response = client.post("/predict", json=payload)
    # Must reject with 422 Unprocessable Entity, not crash with 500
    assert response.status_code == 422

# ---------------------------------------------------------------------------
# Test 4: Input Validation Defense (String instead of Float)
# ---------------------------------------------------------------------------
def test_predict_rejects_non_numeric_charges(client):
    payload = {
        "tenure": 10,
        "MonthlyCharges": "one_hundred_dollars",  # Invalid type
        "TotalCharges": 500.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

# ---------------------------------------------------------------------------
# Test 5: Edge Case - Brand New Customer Boundary (tenure = 0)
# ---------------------------------------------------------------------------
def test_predict_boundary_zero_tenure(client):
    payload = {
        "tenure": 0,
        "MonthlyCharges": 20.0,
        "TotalCharges": 20.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0