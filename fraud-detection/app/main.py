
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


# Paths and configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
THRESHOLD = 0.982

FEATURES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

# Load trained artifacts
model = joblib.load(ARTIFACTS_DIR / "logistic_regression.joblib")
scaler = joblib.load(ARTIFACTS_DIR / "robust_scaler.joblib")

# Application setup
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Predicts fraud probability and flags transactions "
        "for human review."
    ),
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=APP_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=APP_DIR / "templates")


# Request and response schemas
class Transaction(BaseModel):
    Time: float = Field(ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)


class PredictionResponse(BaseModel):
    fraud_probability: float
    predicted_label: str
    review_flag: bool
    threshold: float


# Routes
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "Logistic Regression",
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    # Preserve the exact feature order used during training
    row = pd.DataFrame(
        [transaction.model_dump()],
        columns=FEATURES,
    )

    # Scale only Time and Amount, using the saved scaler
    row[["Time", "Amount"]] = scaler.transform(
        row[["Time", "Amount"]]
    )

    probability = float(model.predict_proba(row)[0, 1])
    flagged = probability >= THRESHOLD

    return PredictionResponse(
        fraud_probability=round(probability, 6),
        predicted_label="fraud" if flagged else "legitimate",
        review_flag=flagged,
        threshold=THRESHOLD,
    )
