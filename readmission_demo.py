"""Runnable 30-day readmission risk demo on synthetic patient encounters.

Public demo only: data is generated in ``make_encounters`` with clinically
plausible risk drivers (prior admissions, length of stay, comorbidities, age,
discharge disposition). No real patient data, no downloads.

Mirrors the production pipeline: feature engineering -> preprocessing pipeline
-> gradient-boosted model -> threshold chosen for recall -> per-request latency
check for real-time scoring.

Usage:
    python readmission_demo.py              # train, evaluate, report latency
    python readmission_demo.py --n 50000    # bigger synthetic cohort
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age", "length_of_stay", "prior_admissions_12m", "num_medications", "num_comorbidities", "hba1c"]
CATEGORICAL = ["discharge_to", "admission_type"]


def make_encounters(n: int = 20000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 95, n),
        "length_of_stay": rng.gamma(2.0, 2.2, n).round().clip(1, 30),
        "prior_admissions_12m": rng.poisson(0.8, n),
        "num_medications": rng.poisson(9, n),
        "num_comorbidities": rng.poisson(2.2, n),
        "hba1c": rng.normal(6.8, 1.4, n).round(1).clip(4.5, 14),
        "discharge_to": rng.choice(["home", "home_health", "skilled_nursing", "rehab"], n, p=[0.6, 0.2, 0.15, 0.05]),
        "admission_type": rng.choice(["emergency", "urgent", "elective"], n, p=[0.55, 0.25, 0.20]),
    })
    logit = (
        -3.2
        + 0.55 * df.prior_admissions_12m
        + 0.06 * df.length_of_stay
        + 0.22 * df.num_comorbidities
        + 0.015 * (df.age - 60)
        + 0.18 * (df.hba1c > 8)
        + 0.03 * df.num_medications
        + df.discharge_to.map({"home": 0, "home_health": 0.3, "skilled_nursing": 0.6, "rehab": 0.4})
        + df.admission_type.map({"emergency": 0.35, "urgent": 0.15, "elective": -0.3})
        + 0.4 * (df.prior_admissions_12m > 1) * (df.num_comorbidities > 3)  # interaction a linear model misses
    )
    df["readmitted_30d"] = rng.random(n) < 1 / (1 + np.exp(-logit))
    return df


def build_model(kind: str = "gbm") -> Pipeline:
    if kind == "logreg":
        pre = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                                 ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        pre = ColumnTransformer([("num", "passthrough", NUMERIC),
                                 ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0,
            early_stopping=True, class_weight="balanced", random_state=0)
    return Pipeline([("pre", pre), ("clf", clf)])


def threshold_for_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float = 0.70) -> float:
    """Highest threshold that still catches ``target_recall`` of readmissions."""
    for t in np.sort(np.unique(scores))[::-1]:
        if recall_score(y_true, scores >= t) >= target_recall:
            return float(t)
    return 0.0


def run(n: int = 20000) -> dict[str, float]:
    df = make_encounters(n)
    X, y = df[NUMERIC + CATEGORICAL], df["readmitted_30d"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=0)

    results: dict[str, float] = {"readmission_rate": float(y.mean())}
    models, val_pr = {}, {}
    for kind in ("logreg", "gbm"):
        models[kind] = build_model(kind).fit(X_tr, y_tr)
        val_pr[kind] = average_precision_score(y_val, models[kind].predict_proba(X_val)[:, 1])
        p = models[kind].predict_proba(X_te)[:, 1]
        results[f"{kind}_test_roc_auc"] = roc_auc_score(y_te, p)
        results[f"{kind}_test_pr_auc"] = average_precision_score(y_te, p)

    # Champion = best validation PR-AUC (test set is only used for the final report).
    champion = max(val_pr, key=val_pr.get)
    model = models[champion]
    results[f"champion_is_{champion}"] = 1.0
    p_te = model.predict_proba(X_te)[:, 1]
    t = threshold_for_recall(y_val, model.predict_proba(X_val)[:, 1])  # tuned on validation, not test
    results["recall_at_threshold"] = recall_score(y_te, p_te >= t)
    results["precision_at_threshold"] = precision_score(y_te, p_te >= t)

    one = X_te.iloc[[0]]
    model.predict_proba(one)  # warm-up
    start = time.perf_counter()
    for _ in range(200):
        model.predict_proba(one)
    results["single_prediction_ms"] = (time.perf_counter() - start) / 200 * 1000
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=20000, help="number of synthetic encounters")
    args = parser.parse_args()
    for name, value in run(args.n).items():
        print(f"{name:>28}: {value:.3f}")


if __name__ == "__main__":
    main()
