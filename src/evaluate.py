import pandas as pd
import joblib
import json
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

test = pd.read_csv("data/processed/test.csv")

target = "Churn"
X_test = test.drop(columns=[target])
y_test = test[target]

model = joblib.load("models/best_model.joblib")

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:,1]

metrics = {
    "accuracy": round(accuracy_score(y_test, preds), 4),
    "f1_score": round(f1_score(y_test, preds), 4),
    "roc_auc": round(roc_auc_score(y_test, probs), 4)
}

os.makedirs("reports", exist_ok=True)
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Evaluation complete!")
print(json.dumps(metrics, indent=4))