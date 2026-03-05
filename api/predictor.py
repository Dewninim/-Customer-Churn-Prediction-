import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
model = joblib.load(MODEL_PATH)

# These are the min/max values from original dataset for scaling
SCALE_RANGES = {
    "tenure":         {"min": 0,    "max": 72},
    "MonthlyCharges": {"min": 18.25,"max": 118.75},
    "TotalCharges":   {"min": 0,    "max": 8684.8}
}

NUMERIC_OR_RAW = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "PaperlessBilling", "MonthlyCharges", "TotalCharges"
]

ONE_HOT_COLS = [
    "InternetService_DSL", "InternetService_Fiber optic", "InternetService_No",
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
    "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"
]

RAW_TO_ONEHOT = {
    "InternetService": [
        "InternetService_DSL",
        "InternetService_Fiber optic",
        "InternetService_No"
    ],
    "Contract": [
        "Contract_Month-to-month",
        "Contract_One year",
        "Contract_Two year"
    ],
    "PaymentMethod": [
        "PaymentMethod_Bank transfer (automatic)",
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed check"
    ]
}

def scale_value(value, col):
    min_val = SCALE_RANGES[col]["min"]
    max_val = SCALE_RANGES[col]["max"]
    return (value - min_val) / (max_val - min_val)

def preprocess_input(features: dict):
    base = {}

    yes_no_fields = [
        "Partner", "Dependents", "PhoneService", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "PaperlessBilling"
    ]

    base["gender"] = 1 if features["gender"].lower() == "male" else 0
    base["SeniorCitizen"] = features["SeniorCitizen"]

    for field in yes_no_fields:
        base[field] = 1 if features[field].lower() == "yes" else 0

    # Scale numeric fields
    base["tenure"] = scale_value(features["tenure"], "tenure")
    base["MonthlyCharges"] = scale_value(features["MonthlyCharges"], "MonthlyCharges")
    base["TotalCharges"] = scale_value(features["TotalCharges"], "TotalCharges")

    # One-hot encode
    for cat_field, possible_cols in RAW_TO_ONEHOT.items():
        val = features[cat_field]
        for colname in possible_cols:
            if val.lower() in colname.lower():
                base[colname] = 1
            else:
                base[colname] = 0

    df = pd.DataFrame([base])
    full_col_list = NUMERIC_OR_RAW + ONE_HOT_COLS
    for col in full_col_list:
        if col not in df.columns:
            df[col] = 0
    df = df[full_col_list]
    return df

def predict(features: dict):
    X = preprocess_input(features)
    prob = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])
    return prob, pred