# Customer Churn Prediction — MLOps Pipeline

> A full end-to-end MLOps project for predicting customer churn using a reproducible pipeline, experiment tracking, automated scheduling, and REST API deployment.
>
> **SLTC Research University | Machine Learning Module — Final Assignment 2026**

---

## 📌 Project Overview

This project builds a production-ready machine learning pipeline to predict customer churn for a telecommunications company. The pipeline covers data ingestion, preprocessing, model training, evaluation, experiment tracking, orchestration, and API deployment — following MLOps best practices throughout.

**Dataset:** Telco Customer Churn (7,043 records, 21 features)  
**Target:** `Churn` (binary: Yes/No)  
**Best Model:** Logistic Regression (GridSearchCV tuned)  
**Metrics:** Accuracy 80.27% | F1 59.94% | ROC-AUC 84.25%  
**Deadline:** March 8th, 11:59 PM

---

## 👥 Team Members & Responsibilities

| Member | Role | Code Task | Report Chapters |
|--------|------|-----------|-----------------|
| Member 1 | Data Engineer | Data cleaning, encoding, scaling, DVC tracking | Ch 3: Dataset, Ch 4: EDA |
| Member 2 | ML Engineer | Train 3 models, MLflow logging, best model | Ch 6: Models, Ch 7: Tuning |
| Member 3 | Pipeline Engineer | dvc.yaml pipeline, dvc repro, DAGsHub push | Ch 5: Features, Ch 8: Architecture, Ch 10: MLflow |
| Member 4 | Airflow Engineer | Airflow DAG, 6 tasks, full run end-to-end | Ch 9: Airflow, Ch 12: Challenges |
| Member 5 | API & Deployment | FastAPI /predict, Dockerfile, API test, compile report | Ch 11: Deployment, Ch 13: Future |
| Member 6 | DevOps + Bonus | DAGsHub setup, README, LLM bonus | Ch 1: Introduction, Ch 2: Problem |

---

## 🗂️ Project Structure

```
-Customer-Churn-Prediction-/
│
├── data/
│   ├── raw/
│   │   └── churn.csv                  # Raw dataset (DVC tracked)
│   └── processed/
│       ├── train.csv                  # Training split 80% (DVC tracked)
│       └── test.csv                   # Test split 20% (DVC tracked)
│
├── src/
│   ├── data_ingestion.py              # Load CSV, fix TotalCharges blanks
│   ├── preprocessing.py               # Encode, scale, split data
│   ├── train.py                       # Train 3 models + MLflow logging
│   └── evaluate.py                    # Load model, print metrics, save plots
│
├── models/
│   ├── best_model.pkl                 # Best model (pickle format)
│   ├── best_model.joblib              # Best model (joblib format)
│   └── scaler.pkl                     # MinMaxScaler for API inference
│
├── reports/
│   ├── metrics.json                   # Final evaluation metrics (DVC tracked)
│   ├── feature_schema.json            # Feature names and types for API
│   ├── feature_importance.png         # Top 10 feature importance chart
│   └── figures/
│       ├── roc_curve.png              # ROC curve plot
│       └── confusion_matrix.png       # Confusion matrix plot
│
├── api/
│   ├── __init__.py
│   ├── main.py                        # FastAPI app with /health and /predict
│   ├── schemas.py                     # Pydantic input/output models
│   └── predictor.py                   # Feature preprocessing + model inference
│
├── airflow_dags/
│   └── churn_dag.py                   # Airflow DAG — 6 task pipeline
│
├── Dockerfile                         # Docker image for API deployment
├── docker-compose.yml                 # Multi-container orchestration
├── dvc.yaml                           # DVC 4-stage pipeline definition
├── dvc.lock                           # DVC pipeline lock file
├── params.yaml                        # Configurable pipeline parameters
├── .dvcignore                         # DVC ignore rules
├── .gitignore                         # Git ignore rules
├── .env.example                       # Environment variable template
├── requirements.txt                   # All Python dependencies
└── README.md                          # This file
```

---

## ⚙️ Pipeline Stages (DVC) — Member 3

The pipeline is defined in `dvc.yaml` with 4 reproducible stages:

```
data/raw/dataset.csv
       │
       ▼
  data_ingestion     →  data/raw/churn.csv
       │
       ▼
  preprocessing      →  data/processed/train.csv
                     →  data/processed/test.csv
                     →  models/scaler.pkl
                     →  reports/feature_schema.json
       │
       ▼
  train              →  models/best_model.joblib
                     →  models/best_model.pkl
       │
       ▼
  evaluate           →  reports/metrics.json
                     →  reports/figures/roc_curve.png
                     →  reports/figures/confusion_matrix.png
```

Reproduce the full pipeline from scratch:
```bash
dvc repro
```

View the pipeline DAG:
```bash
dvc dag
```

---

## 🔵 Member 1 — Data Engineering

**Scripts:** `src/data_ingestion.py`, `src/preprocessing.py`

Responsibilities:
- Load raw CSV and fix `TotalCharges` (convert blanks to NaN, cast to float)
- Encode categorical columns (Yes/No → 0/1, multi-class → one-hot encoding)
- Scale numeric features: `tenure`, `MonthlyCharges`, `TotalCharges` using MinMaxScaler
- Train-test split 80/20 — save as `data/processed/train.csv` and `data/processed/test.csv`
- Track `data/raw/` and `data/processed/` using DVC

Run data pipeline:
```bash
python src/data_ingestion.py
python src/preprocessing.py
```

---

## 🟣 Member 2 — Model Training & Evaluation

**Scripts:** `src/train.py`, `src/evaluate.py`

### Models Trained
- Logistic Regression ✅ (Best model)
- Random Forest
- XGBoost

### Hyperparameter Tuning
- GridSearchCV applied to Logistic Regression
- Parameters tuned: `C`, `solver`, `max_iter`

### Experiment Tracking
- All 3 models logged to MLflow with parameters, metrics, confusion matrix, ROC curve

Run training:
```bash
python src/train.py
```

Run evaluation:
```bash
python src/evaluate.py
```

### Final Metrics (Best Model — Logistic Regression)

| Metric | Score |
|--------|-------|
| Accuracy | 80.27% |
| F1 Score | 59.94% |
| ROC-AUC | 84.25% |

---

## 🟠 Member 3 — DVC Pipeline & DAGsHub

**Files:** `dvc.yaml`, `params.yaml`, `.dvcignore`

Responsibilities:
- Define all 4 pipeline stages in `dvc.yaml` with correct `deps` and `outs`
- Configure `params.yaml` with `test_size`, `random_state`, model parameters
- Set up DVC remote pointing to DAGsHub storage
- Run `dvc repro` to confirm full pipeline executes without errors
- Push all tracked data and models to DAGsHub

Push data to DAGsHub remote:
```bash
dvc push
```

Pull data from DAGsHub remote:
```bash
dvc pull
```

**DAGsHub:** https://dagshub.com/Dewni/-Customer-Churn-Prediction-

---

## 🟡 Member 4 — Airflow Orchestration

**File:** `airflow_dags/churn_dag.py`

The full pipeline is automated as an Airflow DAG with 6 tasks:

| Task | Description |
|------|-------------|
| `data_ingestion` | Loads raw CSV and saves churn.csv |
| `data_validation` | Checks nulls, schema, class balance |
| `feature_engineering` | Runs preprocessing and scaling |
| `model_training` | Trains all 3 models with MLflow logging |
| `model_evaluation` | Evaluates best model and saves metrics |
| `model_registration` | Registers final model to MLflow Model Registry |

Task execution order:
```
data_ingestion >> data_validation >> feature_engineering >> model_training >> model_evaluation >> model_registration
```

Start Airflow:
```bash
airflow standalone
```

Open Airflow UI at http://localhost:8080

---

## 🔴 Member 5 — API Deployment

**Files:** `api/main.py`, `api/schemas.py`, `api/predictor.py`, `Dockerfile`

### Run API locally:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Returns `{"status": "ok"}` |
| POST | `/predict` | Returns churn probability and prediction |

### Example Request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaperlessBilling": "Yes",
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check"
  }'
```

### Example Response:
```json
{
  "churn_probability": 0.5848,
  "prediction": 1
}
```

---

## 🟢 Member 6 — DevOps & Bonus

**Files:** `README.md`, `.gitignore`, `.env.example`, project folder structure

Responsibilities:
- Created DAGsHub repo and initialized Git on Day 1
- Connected MLflow tracking URI to DAGsHub
- Connected DVC remote storage to DAGsHub
- Set up `.gitignore` to exclude `data/`, `models/`, `__pycache__/`, `.env`, `.venv/`
- Established project folder structure for the entire team
- Wrote this README

### BONUS — LLM Retention Message Generator
A script that takes a customer profile + churn prediction and calls the Claude/GPT API to generate a personalized retention message using prompt engineering.

---

## 🐳 Docker Deployment

Build and run with Docker:
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

Run with Docker Compose:
```bash
docker-compose up --build
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- Git
- DVC (`pip install dvc`)
- Docker (optional, for containerised deployment)

### 1. Clone the repository
```bash
git clone https://github.com/Dewninim/-Customer-Churn-Prediction-
cd -Customer-Churn-Prediction-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull DVC tracked data
```bash
dvc pull
```

### 4. Reproduce the full pipeline
```bash
dvc repro
```

### 5. Run the API
```bash
uvicorn api.main:app --reload
```

### 6. View MLflow experiments
```bash
mlflow ui
```
Open http://127.0.0.1:5000

---

## 🔗 Links

- **GitHub:** https://github.com/Dewninim/-Customer-Churn-Prediction-
- **DAGsHub:** https://dagshub.com/Dewni/-Customer-Churn-Prediction-
- **MLflow (DAGsHub):** https://dagshub.com/Dewni/-Customer-Churn-Prediction-.mlflow

---

## 📅 Project Timeline

| Day    | Date    | Milestone |
|-----   |------   |-----------|
| Day 1  | Feb 27  | Member 6: Repo setup. Everyone clones. |
| Day 2  | Feb 28  | Member 1: data scripts. Member 6: README skeleton. |
| Day 3  | Mar 1   | Member 1: DVC tracking. Member 2: model training starts. Member 3: dvc.yaml. |
| Day 4  | Mar 2   | Member 2: All 3 models + MLflow + best_model.pkl. Member 3: dvc repro done. |
| Day 5  | Mar 3   | Member 4: Airflow DAG. Member 5: FastAPI + Docker. Member 6: DAGsHub connected. |
| Day 6  | Mar 4   | Full end-to-end test. Integration bugs fixed. |
| Day 7  | Mar 5   | Member 6: LLM bonus. All chapters submitted to Member 5. |
| Day 8  | Mar 6   | Member 5: Compile full report. Code cleanup. |
| Day 9  | Mar 7   | Code comments, final DVC push, final Git commit, video demo. |
| Day 10 | Mar 8   | **Final submission by 11:59 PM** |
