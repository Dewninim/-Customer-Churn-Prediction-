import pandas as pd
import os

SOURCE_PATH = "data/raw/dataset.csv"
RAW_OUTPUT = "data/raw/churn.csv"

def ingest_data():
    print("Starting data ingestion...")
    if not os.path.exists(SOURCE_PATH):
        raise FileNotFoundError("Source dataset not found")
    df = pd.read_csv(SOURCE_PATH)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(RAW_OUTPUT, index=False)
    print(f"Raw dataset saved → {RAW_OUTPUT}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    ingest_data()