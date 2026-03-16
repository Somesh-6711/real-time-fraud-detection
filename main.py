from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    title="Real-Time Fraud Detection API",
    version="1.0.0",
    description="Production-style fraud detection service using ML pipelines, threshold tuning, and XGBoost inference."
)

app.include_router(router)