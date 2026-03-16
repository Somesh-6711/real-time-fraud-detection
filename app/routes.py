from fastapi import APIRouter
from app.schemas import FraudRequest, FraudResponse
from src.pipeline.prediction_pipeline import PredictionPipeline

router = APIRouter()
pipeline = PredictionPipeline()


@router.get("/health")
def health_check():
    return {"status": "ok", "message": "Fraud detection API is running"}


@router.post("/predict", response_model=FraudResponse)
def predict_fraud(request: FraudRequest):
    result = pipeline.predict(request.model_dump())
    return FraudResponse(**result)

from src.utils.common import read_yaml

@router.get("/metadata")
def model_metadata():
    threshold_report = read_yaml("artifacts/model_trainer/threshold_report.yaml")

    return {
        "model_name": "XGBoost Fraud Detection Model",
        "threshold": threshold_report["best_threshold"],
        "problem_type": "Binary Classification",
        "target": "Fraud Detection",
        "features": 33
    }