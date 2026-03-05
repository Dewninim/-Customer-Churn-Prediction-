from fastapi import FastAPI
from api.schemas import PredictInput, PredictOutput
from api.predictor import predict

app = FastAPI(title="Customer Churn Prediction API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictOutput)
def predict_api(input: PredictInput):
    try:
        proba, pred = predict(input.dict())
        return PredictOutput(churn_probability=proba, prediction=pred)
    except Exception as e:
        return {"churn_probability": 0.0, "prediction": 0}