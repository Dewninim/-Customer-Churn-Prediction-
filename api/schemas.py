from pydantic import BaseModel


class PredictInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str

    InternetService: str
    Contract: str
    PaymentMethod: str

    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    PaperlessBilling: str
    MonthlyCharges: float
    TotalCharges: float


class PredictOutput(BaseModel):
    churn_probability: str   # keep string if you want "0.0000" formatting
    prediction: int
    message: str


