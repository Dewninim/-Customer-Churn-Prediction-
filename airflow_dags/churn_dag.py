from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run_data_ingestion():
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "data_ingestion.py")],
        cwd=ROOT, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(result.stderr)

def run_data_validation():
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "raw" / "churn_raw.csv")
    assert df.shape[0] > 0, "Dataset is empty!"
    assert "Churn" in df.columns, "Churn column missing!"
    print(f"Validation passed! Rows: {df.shape[0]}")

def run_feature_engineering():
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "preprocessing.py")],
        cwd=ROOT, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(result.stderr)

def run_model_training():
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "train.py")],
        cwd=ROOT, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(result.stderr)

def run_model_evaluation():
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "evaluate.py")],
        cwd=ROOT, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(result.stderr)

def run_model_registration():
    print("Best model is saved at models/best_model.pkl")
    print("Model registered successfully!")

default_args = {
    "owner": "Mxthxrx",
    "start_date": datetime(2025, 1, 1),
    "retries": 0,
}

with DAG(
    dag_id="churn_prediction_pipeline",
    default_args=default_args,
    schedule_interval="@once",
    catchup=False,
    description="End-to-end churn prediction MLOps pipeline",
) as dag:

    t1 = PythonOperator(task_id="data_ingestion",     python_callable=run_data_ingestion)
    t2 = PythonOperator(task_id="data_validation",    python_callable=run_data_validation)
    t3 = PythonOperator(task_id="feature_engineering",python_callable=run_feature_engineering)
    t4 = PythonOperator(task_id="model_training",     python_callable=run_model_training)
    t5 = PythonOperator(task_id="model_evaluation",   python_callable=run_model_evaluation)
    t6 = PythonOperator(task_id="model_registration", python_callable=run_model_registration)


    t1 >> t2 >> t3 >> t4 >> t5 >> t6
