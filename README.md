# MLOps Pipeline for Model Deployment and Monitoring

## Project Overview

In production machine learning systems, infrastructure health (HTTP 200 OK) does not guarantee predictive reliability. Models often suffer from silent performance degradation due to Covariate Shift—where live input data distributions drift from the original training baseline.

This project implements an end-to-end, decoupled MLOps pipeline for a Telecom Customer Churn prediction model. It features robust continuous integration, strict API schema validation, and an automated statistical monitoring engine that detects unlabelled data drift and triggers a model retraining and recovery workflow.

## System Architecture and Technology Stack

The system is built on a decoupled architecture, ensuring that data, experimentation, serving, and monitoring can scale and be tested independently.

* **Data Versioning:** DVC (Data Version Control)
* **Experiment Tracking & Registry:** MLflow (SQLite backend)
* **Model Serving:** FastAPI, Uvicorn, Pydantic (Strict schema boundary validation)
* **Containerization:** Docker
* **CI/CD Automation:** GitHub Actions
* **Drift Monitoring:** Two-sample Kolmogorov-Smirnov (KS) Test & Population Stability Index (PSI)
* **Testing & QA:** Pytest (8-test suite) & Flake8 (Static syntax analysis)

## Key Features and Quantitative Results

* **Strict Inference Boundaries:** Uses Pydantic to instantly reject invalid inference payloads (e.g., negative tenure, non-numeric charges) with HTTP 422 errors, protecting the ML model from malformed inputs.
* **Statistical Drift Auditor:** Continuously audits buffered live telemetry against the training baseline.
  * *KS Test:* Detects fine-grained continuous shifts (Alerts if $p < 0.05$).
  * *PSI:* Quantifies binned distributional shifts (Alerts if $PSI > 0.20$).
* **Automated Retraining Recovery:** During a controlled +35% billing distortion stress test, the baseline model's accuracy dropped from 78.68% to 74.48%. The pipeline successfully detected the covariate shift and triggered a retraining loop, recovering the model's accuracy to 80.31% (+5.83% improvement over the degraded state).

## Repository Structure

```text
├── .github/workflows/ci.yaml   # GitHub Actions CI/CD pipeline
├── data/
│   ├── churn.csv.dvc           # DVC pointer for the dataset
│   └── .gitignore              
├── reports/
│   ├── drift_audit_report.json # Generated KS and PSI statistical audits
│   └── recovery_benchmark.json # Pre and post-retraining metrics
├── src/
│   ├── train.py                # Model training and MLflow registration
│   ├── serve.py                # FastAPI inference and Pydantic schemas
│   ├── monitor.py              # KS-Test & PSI drift detection engine
│   └── simulate_drift.py       # Stress testing & continuous training trigger
├── tests/
│   ├── test_serve.py           # API schema and boundary unit tests
│   └── test_monitor.py         # Statistical calculation unit tests
├── Dockerfile                  # Container definition for the FastAPI service
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies
└── README.md                   
```

## Local Setup and Execution

### 1. Environment Initialization

Clone the repository and set up the virtual environment:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Quality Assurance and Testing

Run the Flake8 static code analysis to check for syntax and undefined-name errors:

```bash
flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
```

Execute the automated test suite (API and Monitoring components):

```bash
pytest -v
```

### 3. Drift Stress Test and Retraining Simulation

Execute the core MLOps workflow to simulate a +35% billing data drift, generate the audit reports, and trigger the recovery pipeline:

```bash
python -m src.simulate_drift
```

### 4. View Experiment Ledger

Start the MLflow UI to view the baseline and retrained model artifacts, parameters, and metrics:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --workers 1
```

Navigate to `http://localhost:5000` in your browser.

### 5. Containerization

Build the production-ready Docker image for the inference service:

```bash
docker build -t mlops-churn:latest .
```

## CI/CD Pipeline Workflow

Every push to the main branch triggers the GitHub Actions workflow (`ci.yaml`), which executes the following sequence:

1. **Environment Setup:** Installs Python 3.11 and required dependencies.
2. **Static Analysis:** Enforces code quality via Flake8.
3. **Data Materialization:** Pulls the versioned benchmark dataset via DVC.
4. **Baseline Training:** Executes MLflow training to establish reference metrics.
5. **Synthetic Drift Evaluation:** Validates the drift detection and recovery execution logic.
6. **Automated Testing:** Runs the Pytest suite as a serving gate.
7. **Docker Build:** Packages the verified application into a container image.

## Future Roadmap

* **Kubernetes Scalability:** Deploy the serving layer to a K8s cluster with Horizontal Pod Autoscaling (HPA) and canary traffic splitting.
* **Concept Drift P(Y|X):** Implement delayed ground-truth label ingestion to monitor changes in the feature-target relationship.
* **Explainable AI (XAI):** Integrate TreeSHAP to provide feature contribution attribution for individual API predictions.