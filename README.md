# Patient Readmission & Disease Risk Prediction

**Repo name on GitHub:** `patient-readmission-prediction`
**Context:** Machine Learning Engineer @ Kaiser Permanente (Feb 2024 – Present)

> Case study of production work. This repo describes the system design and results — it contains no employer code, data, or patient information.

## What it is
Machine learning models predicting patient readmission risk and disease risk from structured clinical and claims data, deployed as production inference services.

## What I built
- Preprocessing pipelines (missing values, encoding, normalization) for 5M+ patient records
- Healthcare features engineered from 3+ years of demographic, diagnosis, medication, and encounter data
- TensorFlow deep learning models for disease risk prediction
- FastAPI + Docker inference services and automated SageMaker batch prediction pipelines
- MLflow experiment tracking (100+ iterations), drift monitoring, and retraining strategy

## Results
- **+15%** readmission prediction accuracy
- **+12%** disease risk prediction accuracy over baseline
- **<100ms** inference response times
- **>90% accuracy maintained over 6 months** with drift monitoring and retraining
- **15 hours/week saved** through automated batch pipelines

## Tech stack
Python · TensorFlow · Pandas · NumPy · AWS SageMaker · MLflow · FastAPI · Docker · PostgreSQL · Power BI

## Skills demonstrated
Predictive modeling · Feature engineering at scale · Deep learning · MLOps · Model monitoring / drift detection
