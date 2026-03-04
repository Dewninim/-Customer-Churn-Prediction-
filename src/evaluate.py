import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

test = pd.read_csv("data/processed/test.csv")

target = "Churn"

X_test = test.drop(columns=[target])
y_test = test[target]

model = joblib.load("models/best_model.joblib")

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, preds))
print("F1 Score:", f1_score(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, probs))
