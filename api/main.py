from fastapi import FastAPI, HTTPException
from api.schemas import PredictInput, PredictOutput
from api.predictor import predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Customer Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production: replace * with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOutput)
def predict_api(payload: PredictInput):
    try:
        proba, pred = predict(payload.model_dump())

        churn_probability = f"{float(proba):.2f}"

        message = (
            "Customer likely to churn. Consider a retention offer or follow-up."
            if int(pred) == 1
            else "Customer likely to stay. No immediate retention action needed."
        )

        return PredictOutput(
            churn_probability=churn_probability,
            prediction=int(pred),
            message=message,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))