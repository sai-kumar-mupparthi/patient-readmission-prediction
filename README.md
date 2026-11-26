# Patient Readmission & Disease Risk Prediction

**Repo name on GitHub:** `patient-readmission-prediction`
**Context:** Machine Learning Engineer @ Kaiser Permanente (Feb 2024 – Present)

> Case study of production work. This repo describes the system design and results — it contains no employer code, data, or patient information.

> ▶ **Includes a runnable demo on synthetic data:** see [Run the demo](#run-the-demo) below.

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

---

## Run the demo

[![tests](https://github.com/sai-kumar-mupparthi/patient-readmission-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/sai-kumar-mupparthi/patient-readmission-prediction/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

> **About this demo.** The production system above is described as a case study and contains no employer code or data. This repo also includes a small, runnable demo that applies the same approach at laptop scale on **synthetic / sample data**. Its numbers come from that synthetic data and are not the production results.

```bash
pip install -r requirements-demo.txt
python readmission_demo.py
pytest -q
```

**Sample output**

```text
$ python readmission_demo.py
            readmission_rate: 0.219
         logreg_test_roc_auc: 0.715
          logreg_test_pr_auc: 0.433
            gbm_test_roc_auc: 0.709
             gbm_test_pr_auc: 0.423
          champion_is_logreg: 1.000
         recall_at_threshold: 0.710
      precision_at_threshold: 0.328
        single_prediction_ms: 2.101
```

**What's inside `readmission_demo.py`**

- A synthetic cohort of 20,000 encounters with plausible risk drivers (prior admissions, length of stay, comorbidities, HbA1c, discharge disposition)
- scikit-learn `Pipeline` + `ColumnTransformer` preprocessing
- Logistic regression vs. gradient boosting, with the champion picked on **validation** PR-AUC (the test set is only used for the final report)
- A decision threshold tuned for ~70% recall, since missing a readmission costs more than an extra follow-up call
- A single-prediction latency check for real-time scoring
