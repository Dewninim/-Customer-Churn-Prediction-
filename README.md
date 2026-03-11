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

## ⚙️ Pipeline Stages (DVC)

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

### Responsibilities
- Load raw CSV and fix `TotalCharges` column (convert blank strings to NaN, cast to float)
- Encode categorical columns: Yes/No fields → 0/1 binary encoding
- One-hot encode multi-class columns: `InternetService`, `Contract`, `PaymentMethod`
- Scale numeric features: `tenure`, `MonthlyCharges`, `TotalCharges` using MinMaxScaler
- Save scaler as `models/scaler.pkl` for use by the API
- Train-test split 80/20 — save as `data/processed/train.csv` and `data/processed/test.csv`
- Track `data/raw/` and `data/processed/` using DVC (`dvc add`)
- Share exact column names and feature schema with Members 2 and 5

### Run Data Pipeline
```bash
python src/data_ingestion.py
python src/preprocessing.py
```

### Output Files
- `data/raw/churn.csv` — cleaned raw data
- `data/processed/train.csv` — 80% training split
- `data/processed/test.csv` — 20% test split
- `models/scaler.pkl` — fitted MinMaxScaler
- `reports/feature_schema.json` — feature column names and types

### Report Chapters
- **Chapter 3 — Dataset Description:** All 21 columns, data types, class imbalance in Churn column, summary statistics table
- **Chapter 4 — EDA:** Churn distribution chart, correlation heatmap, churn by Contract type, churn by tenure, MonthlyCharges box plot (minimum 7–8 visuals with explanations)

---

## 🟣 Member 2 — Model Training & Evaluation

**Scripts:** `src/train.py`, `src/evaluate.py`

### Responsibilities
- Train three machine learning models on Member 1's preprocessed data
- Evaluate all 3 models: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Set up MLflow experiment tracking: log parameters, metrics, confusion matrix image, ROC curve image per model
- Run hyperparameter tuning on the best model using GridSearchCV
- Save the final best model as `models/best_model.pkl` and `models/best_model.joblib`
- Write standalone `evaluate.py` that loads the saved model and prints all metrics

### Models Trained
| Model | Description |
|-------|-------------|
| Logistic Regression | Linear classifier, tuned with GridSearchCV — **Best Model** |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosting classifier |

### Hyperparameter Tuning
- Method: GridSearchCV with cross-validation
- Applied to: Logistic Regression (best performing model)
- Parameters tuned: `C`, `solver`, `max_iter`, `class_weight`

### MLflow Experiment Tracking
- Experiment name: `churn_training_pipeline`
- Logged per run: model type, hyperparameters, accuracy, F1, ROC-AUC, confusion matrix image, ROC curve image
- View all runs: `mlflow ui` → http://127.0.0.1:5000

### Run Training
```bash
python src/train.py
```

### Run Evaluation
```bash
python src/evaluate.py
```

### Final Metrics — Best Model (Logistic Regression)

| Metric | Score |
|--------|-------|
| Accuracy | 80.27% |
| F1 Score | 59.94% |
| ROC-AUC | 84.25% |

### Output Files
- `models/best_model.pkl` — saved best model (used by FastAPI)
- `models/best_model.joblib` — saved best model (joblib format)
- `reports/figures/roc_curve.png` — ROC curve plot
- `reports/figures/confusion_matrix.png` — confusion matrix plot
- `reports/metrics.json` — final evaluation metrics

### Report Chapters
- **Chapter 6 — Model Comparison:** Table comparing all 3 models on all metrics, confusion matrix images, ROC curve plots, explanation of which model won and why
- **Chapter 7 — Hyperparameter Tuning:** Parameters tuned, before/after performance table, best parameter values found

---

## 🟠 Member 3 — DVC Pipeline & DAGsHub

**Files:** `dvc.yaml`, `params.yaml`, `.dvcignore`

### Responsibilities
- Write `dvc.yaml` with all 4 stages, each with correct `deps` and `outs` defined
- Write `params.yaml` with configurable values: `test_size`, `random_state`, model parameters
- Run `dvc repro` from scratch and confirm full pipeline completes without errors
- Configure DVC remote to DAGsHub: `dvc remote add` + `dvc push`
- Coordinate with Members 1 and 2 to get correct script paths and output file names

### Push Data to DAGsHub
```bash
dvc push
```

### Pull Data from DAGsHub
```bash
dvc pull
```

### Report Chapters
- **Chapter 5 — Feature Engineering:** Encoding decisions, scaling rationale, feature importance table from best model
- **Chapter 8 — MLOps Architecture:** Full system diagram showing Git → DVC → MLflow → Airflow → FastAPI → DAGsHub data flow
- **Chapter 10 — MLflow Analysis:** Screenshots of MLflow UI, experiment comparison table, explanation of best run

---

## 🟡 Member 4 — Airflow Orchestration

**File:** `airflow_dags/churn_dag.py`

### Responsibilities
- Create Airflow DAG with 6 tasks using PythonOperator
- Set correct task dependencies and execution order
- Run full DAG end-to-end and verify all tasks complete successfully

### DAG Tasks

| Task | Description |
|------|-------------|
| `data_ingestion` | Calls Member 1's ingestion function — loads and saves raw churn.csv |
| `data_validation` | Checks for nulls, validates schema, checks class balance |
| `feature_engineering` | Calls Member 1's preprocessing function — encodes, scales, splits |
| `model_training` | Calls Member 2's train function — trains all 3 models with MLflow |
| `model_evaluation` | Calls Member 2's evaluate function — saves metrics and plots |
| `model_registration` | Logs final best model to MLflow Model Registry |

### Task Execution Order
```
data_ingestion >> data_validation >> feature_engineering >> model_training >> model_evaluation >> model_registration
```

### Start Airflow
```bash
airflow standalone
```

Open Airflow UI at: http://localhost:8080

### Report Chapters
- **Chapter 9 — Airflow DAG:** Explanation of each of the 6 tasks, Airflow UI screenshot showing DAG graph and successful run
- **Chapter 12 — Challenges & Lessons Learned:** Real team challenges during development, integration issues, what was harder than expected, what the team would do differently

---

## 🔴 Member 5 — API Deployment & Report Compilation

**Files:** `api/main.py`, `api/schemas.py`, `api/predictor.py`, `Dockerfile`, `docker-compose.yml`

### Responsibilities
- Build FastAPI app with POST `/predict` endpoint
- Load `models/best_model.pkl` saved by Member 2 for inference
- Apply the same preprocessing (scaler) used during training before prediction
- Write Dockerfile to containerise the API
- Collect all chapters from Members 1–6 by March 6th and compile into final Word document

### Run API Locally
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Returns `{"status": "ok"}` |
| POST | `/predict` | Returns churn probability and prediction |

### Example Request
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

### Example Response
```json
{
  "churn_probability": 0.5848,
  "prediction": 1
}
```

### Report Chapters
- **Chapter 11 — Deployment:** API endpoint explanation, request/response format, Dockerfile walkthrough, screenshot of live API call from curl or Postman
- **Chapter 13 — Future Improvements:** CI/CD pipeline, model monitoring (data drift detection), automated retraining triggers, Kubernetes scaling, business dashboard idea
- **Compile Full Report:** Collect all chapters, merge into one Word document, add cover page, table of contents, consistent fonts/formatting, page numbers — final submission ready

---

## 🟢 Member 6 — DevOps & Bonus

**Files:** `README.md`, `.gitignore`, `.env.example`, project folder structure

### Responsibilities
- **Day 1 Priority:** Create DAGsHub repo, initialise Git, share repo URL with all members
- Connect MLflow tracking URI to DAGsHub so Member 2's experiments appear there
- Connect DVC remote storage to DAGsHub so Member 3's pipeline data appears there
- Set up `.gitignore` to exclude: `data/`, `models/`, `__pycache__/`, `.env`, `.venv/`
- Establish project folder structure for the entire team from day one
- Write and maintain `README.md`

### BONUS — LLM Retention Message Generator
A script that takes a customer profile and churn prediction result, then calls the Claude or GPT API to generate a personalised retention message using prompt engineering. Demonstrates 3 example outputs with different customer types.

### Report Chapters
- **Chapter 1 — Introduction:** Background on telecom churn problem, why ML matters here, brief overview of the solution and tools used
- **Chapter 2 — Problem Definition:** Formal definition of churn, target variable explanation, business goal, success metrics, scope and assumptions

---

## 🐳 Docker Deployment

### Build and Run with Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

### Run with Docker Compose
```bash
docker-compose up --build
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- Git
- DVC (`pip install dvc`)
- Docker (optional)

### 1. Clone the Repository
```bash
git clone https://github.com/Dewninim/-Customer-Churn-Prediction-
cd -Customer-Churn-Prediction-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull DVC Tracked Data
```bash
dvc pull
```

### 4. Reproduce the Full Pipeline
```bash
dvc repro
```

### 5. Run the API
```bash
uvicorn api.main:app --reload
```

### 6. View MLflow Experiments
```bash
mlflow ui
```
Open http://127.0.0.1:5000

## 📈 Final Results

| Metric | Score |
|--------|-------|
| Accuracy | 80.27% |
| F1 Score | 59.94% |
| ROC-AUC | 84.25% |


## 📅 Project Timeline

| Day | Date | Milestone |
|-----|------|-----------|
| Day 1 | Feb 27 | Member 6: Repo setup. Everyone clones. |
| Day 2 | Feb 28 | Member 1: data scripts. Member 6: README skeleton. |
| Day 3 | Mar 1 | Member 1: DVC tracking done. Member 2: model training starts. Member 3: dvc.yaml started. |
| Day 4 | Mar 2 | Member 2: All 3 models + MLflow + best_model.pkl done. Member 3: dvc repro complete. |
| Day 5 | Mar 3 | Member 4: Airflow DAG done. Member 5: FastAPI + Docker done. Member 6: DAGsHub connected. |
| Day 6 | Mar 4 | Full end-to-end test. Integration bugs fixed. All MLflow experiments visible on DAGsHub. |
| Day 7 | Mar 5 | Member 6: LLM bonus. All report chapters submitted to Member 5. |
| Day 8 | Mar 6 | Member 5: Compile full report. Everyone reviews their chapter. Final code cleanup. |
| Day 9 | Mar 7 | Code comments, final DVC push, final Git commit, record video demo. |
| Day 10 | Mar 8 | **Final submission by 11:59 PM** |

## 🔗 Links

- **GitHub:** https://github.com/Dewninim/-Customer-Churn-Prediction-
- **DAGsHub:** https://dagshub.com/Dewni/-Customer-Churn-Prediction-
- **MLflow (DAGsHub):** https://dagshub.com/Dewni/-Customer-Churn-Prediction-.mlflow
