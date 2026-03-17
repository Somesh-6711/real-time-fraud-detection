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
```
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
```

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
```
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
```


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
```
-GET /health
-GET /metadata
-POST /predict
```
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

---

## MLflow Experiment Tracking

The training pipeline uses **MLflow** to track model experiments, parameters, metrics, and artifacts across multiple fraud detection models.

### What is tracked
- Model name
- Hyperparameters
- Evaluation metrics:
  - ROC-AUC
  - PR-AUC
  - Precision
  - Recall
  - F1-score
- Serialized model artifacts

### Why it matters
MLflow improves reproducibility and makes it easier to compare different models and training runs in a production-style workflow.

### Run training with MLflow logging
```
python run_training.py
```

Launch mlflow ui
```
mlflow ui
```
then open
```
http://127.0.0.1:5000
```
Example use in this project

The system logs experiments for:

Logistic Regression

Random Forest

XGBoost

LightGBM

The best model is selected based on PR-AUC, which is more appropriate than accuracy for highly imbalanced fraud detection.

---

Data Drift Monitoring

The project includes data drift monitoring using Evidently to compare reference data and current data distributions.

What is monitored

Distribution changes across all numerical input features

Dataset-level drift summary

Per-feature drift detection scores

Why it matters

In production ML systems, model performance can degrade when incoming data changes over time.
Drift monitoring helps detect these changes early and supports more reliable model behavior in deployment.

Run drift monitoring
```
python run_drift_monitoring.py
```

Output

The drift analysis is saved as:
```
artifacts/monitoring/drift_report.json
```
Baseline result in this project

The initial drift report compared training and test distributions and found:

33 columns checked

0 drifted columns

dataset_drift = false

This indicates no significant distribution shift in the baseline evaluation split.

----


📷 Screenshots

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

