# Real-Time Fraud Detection System

An end-to-end machine learning system for detecting fraudulent financial transactions.  
This project demonstrates a production-style ML pipeline including data validation, feature engineering, model training, threshold tuning, and deployment via a FastAPI inference service.

The system processes transaction features and predicts the probability of fraud in real time through REST APIs.

---

# Project Overview

Financial fraud detection is a highly imbalanced classification problem where the cost of false negatives and false positives must be carefully balanced.  

This project implements a production-ready ML workflow that:

- Validates incoming datasets against schema definitions
- Performs reproducible feature engineering
- Trains and compares multiple ML models
- Selects the best model using PR-AUC (best metric for imbalanced datasets)
- Tunes classification threshold to optimize fraud detection trade-offs
- Exposes the trained model through a FastAPI service
- Containerizes the service using Docker for portable deployment

---

# Dataset

Dataset used: **Credit Card Fraud Detection Dataset**

- Transactions: **284,807**
- Fraud cases: **492**
- Fraud ratio: **0.17%**
- Features: 30 anonymized PCA features + transaction metadata

Target variable:

Class (0 = Normal Transaction, 1 = Fraud)

---

# System Architecture

Raw Data
│
├── Data Ingestion
│
├── Data Validation (Schema Check)
│
├── Feature Engineering
│
├── Train/Test Split
│
├── Model Training
│ ├── Logistic Regression
│ ├── Random Forest
│ ├── XGBoost
│ └── LightGBM
│
├── Model Selection (PR-AUC)
│
├── Threshold Optimization
│
├── Model Artifacts
│ ├── trained model
│ ├── preprocessing pipeline
│ └── evaluation reports
│
└── FastAPI Inference Service


---

# Key Features

• End-to-end ML pipeline with modular architecture  
• Imbalanced classification optimization using PR-AUC  
• Automated threshold tuning for fraud decision boundaries  
• Serialized preprocessing and model artifacts  
• REST API for real-time fraud predictions  
• Docker containerization for reproducible deployment  

---

# Model Performance

Best model selected: **XGBoost**

Evaluation metrics:

| Metric | Score |
|------|------|
| ROC-AUC | 0.98 |
| PR-AUC | 0.88 |
| Precision | 0.93 |
| Recall | 0.81 |
| F1-score | 0.86 |

Confusion Matrix (Test Set)

| | Predicted Normal | Predicted Fraud |
|---|---|---|
| Actual Normal | 56858 | 6 |
| Actual Fraud | 19 | 79 |

---

# Project Structure

real-time-fraud-detection/
│
├── app/ # FastAPI service
│ ├── routes.py
│ └── schemas.py
│
├── artifacts/ # Saved models & reports
│
├── configs/
│ ├── config.yaml
│ ├── params.yaml
│ └── schema.yaml
│
├── data/
│ └── raw/creditcard.csv
│
├── src/
│ ├── components/ # Pipeline stages
│ │ ├── data_ingestion.py
│ │ ├── data_validation.py
│ │ ├── data_transformation.py
│ │ ├── model_trainer.py
│ │ └── model_evaluation.py
│
│ ├── pipeline/
│ │ ├── prediction_pipeline.py
│ │ └── training_pipeline.py
│
│ ├── utils/
│ ├── logger/
│ └── exception/
│
├── Dockerfile
├── requirements.txt
├── run_train



---

# Running the Training Pipeline

Run the full training workflow:

```
python run_training.py
```

This will generate:

- trained model
- preprocessing pipeline
- threshold tuning results
- evaluation report


---

# Running the API

Start the FastAPI server:

Available endpoints:

-GET /health
-GET /metadata
-POST /predict

---

Tech Stack

Python
Scikit-learn
XGBoost
LightGBM
FastAPI
Docker
Pandas
NumPy
Pydantic


📷 Screenshots

<p align="center">
  <img src="images/Screenshot 2025-12-29 025133.png"
    >
</p>
<p align="center">
  <img src="images/Screenshot 2026-03-16 145320.png"
    >
</p>
<p align="center">
  <img src="images/Screenshot 2026-03-16 155032.png"
    >
</p>
<p align="center">
  <img src="images/Screenshot 2026-03-16 155133.png"
    >
</p>

