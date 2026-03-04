import pandas as pd
import mlflow
import mlflow.sklearn
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

target = "Churn"

X_train = train.drop(columns=[target])
y_train = train[target]

X_test = test.drop(columns=[target])
y_test = test[target]

mlflow.set_experiment("churn_training_pipeline")

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs"]
}

model = LogisticRegression(max_iter=5000)

grid = GridSearchCV(model, param_grid, scoring="roc_auc", cv=5)

with mlflow.start_run():

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    probs = best_model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, probs)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.joblib")

print("Training complete")
