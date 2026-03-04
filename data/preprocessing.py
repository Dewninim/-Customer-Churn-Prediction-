import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


INPUT_PATH = "../data/raw/churn.csv"
TRAIN_OUTPUT = "../data/processed/train.csv"
TEST_OUTPUT = "../data/processed/test.csv"
SCHEMA_OUTPUT = "../reports/feature_schema.json"


def preprocess():

    df = pd.read_csv(INPUT_PATH)

    # Drop identifier
    df = df.drop(columns=["customerID"])

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Binary encoding
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    binary_cols = ["Partner","Dependents","PhoneService","PaperlessBilling","Churn"]

    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    service_cols = [
        "MultipleLines","OnlineSecurity","OnlineBackup",
        "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"
    ]

    for col in service_cols:
        df[col] = df[col].apply(lambda x: 1 if x == "Yes" else 0)

    # One hot encoding
    df = pd.get_dummies(
        df,
        columns=["InternetService","Contract","PaymentMethod"]
    )

    # Scaling
    scaler = MinMaxScaler()
    numeric_cols = ["tenure","MonthlyCharges","TotalCharges"]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Train test split
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    train.to_csv(TRAIN_OUTPUT, index=False)
    test.to_csv(TEST_OUTPUT, index=False)

    # Save feature schema
    schema = {
        "features": list(X.columns),
        "target": "Churn"
    }

    with open(SCHEMA_OUTPUT, "w") as f:
        json.dump(schema, f, indent=4)

    print("Preprocessing complete")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")


if __name__ == "__main__":
    preprocess()