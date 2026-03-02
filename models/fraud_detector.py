"""
Fraud Detection Engine
=======================
Trains a Random Forest classifier on claims + policy features to predict fraud.
Includes:
  - Feature engineering (merging claims & policies)
  - Model training with class-weight balancing
  - Evaluation metrics (accuracy, precision, recall, F1, AUC-ROC)
  - Feature importance for explainability
  - Serialization via joblib

Usage:
    python models/fraud_detector.py          # train & save
    from models.fraud_detector import ...    # import functions
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)

# Paths (relative to project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "trained")


# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------
FEATURE_COLS = [
    "age", "credit_score", "annual_premium", "years_as_customer",
    "num_prior_claims", "has_violations", "claim_amount",
    "num_witnesses", "police_report_filed", "report_delay_days",
    "claim_type_enc", "severity_enc", "policy_type_enc", "gender_enc",
]


def load_and_merge():
    """Load CSVs and merge on policy_id, clean numerics."""
    claims = pd.read_csv(os.path.join(DATA_DIR, "claims.csv"))
    policies = pd.read_csv(os.path.join(DATA_DIR, "policies.csv"))
    df = claims.merge(policies, on="policy_id", how="left")

    # Clean claim_amount (some intentionally blank for DQ testing)
    df["claim_amount"] = pd.to_numeric(df["claim_amount"], errors="coerce")
    df.dropna(subset=["claim_amount", "is_fraud"], inplace=True)
    df["is_fraud"] = df["is_fraud"].astype(int)
    return df


def encode_features(df):
    """Label-encode categorical columns; return df + encoder dict."""
    encoders = {}
    for col, src in [
        ("claim_type_enc", "claim_type"),
        ("severity_enc", "severity"),
        ("policy_type_enc", "policy_type"),
        ("gender_enc", "gender"),
    ]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[src].astype(str).str.strip().str.title())
        encoders[src] = le
    return df, encoders


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def train_fraud_model():
    """Train RF + Isolation Forest, save artifacts, return metrics."""
    print("Loading & merging data...")
    df = load_and_merge()
    df, encoders = encode_features(df)

    X = df[FEATURE_COLS].values
    y = df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Random Forest (supervised) ---
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    # Feature importances
    importances = dict(zip(FEATURE_COLS, [round(float(x), 4) for x in rf.feature_importances_]))

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {metrics['roc_auc']}")

    # --- Isolation Forest (unsupervised anomaly detector) ---
    print("\nTraining Isolation Forest (anomaly detection)...")
    iso = IsolationForest(
        n_estimators=150,
        contamination=0.12,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train)

    # --- Save everything ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(MODEL_DIR, "fraud_rf.joblib"))
    joblib.dump(iso, os.path.join(MODEL_DIR, "fraud_iso.joblib"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "label_encoders.joblib"))

    eval_results = {
        "metrics": metrics,
        "confusion_matrix": cm,
        "feature_importances": importances,
        "classification_report": report,
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }
    with open(os.path.join(MODEL_DIR, "evaluation.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nModels saved to {MODEL_DIR}")
    return eval_results


# -------------------------------------------------------------------
# Inference helpers (used by the API)
# -------------------------------------------------------------------
def load_models():
    """Load trained models and encoders."""
    rf = joblib.load(os.path.join(MODEL_DIR, "fraud_rf.joblib"))
    iso = joblib.load(os.path.join(MODEL_DIR, "fraud_iso.joblib"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.joblib"))
    with open(os.path.join(MODEL_DIR, "evaluation.json")) as f:
        evaluation = json.load(f)
    return rf, iso, encoders, evaluation


def predict_fraud(rf, iso, encoders, claim_data: dict):
    """
    Score a single claim. Returns dict with:
      fraud_probability, risk_score, anomaly_flag, risk_level, explanations
    """
    # Encode categoricals
    ct = encoders["claim_type"].transform([str(claim_data.get("claim_type", "Auto")).strip().title()])[0]
    sv = encoders["severity"].transform([str(claim_data.get("severity", "Medium")).strip().title()])[0]
    pt = encoders["policy_type"].transform([str(claim_data.get("policy_type", "Standard")).strip().title()])[0]
    gn = encoders["gender"].transform([str(claim_data.get("gender", "Male")).strip().title()])[0]

    features = np.array([[
        claim_data.get("age", 35),
        claim_data.get("credit_score", 650),
        claim_data.get("annual_premium", 2000),
        claim_data.get("years_as_customer", 3),
        claim_data.get("num_prior_claims", 1),
        claim_data.get("has_violations", 0),
        claim_data.get("claim_amount", 5000),
        claim_data.get("num_witnesses", 1),
        claim_data.get("police_report_filed", 1),
        claim_data.get("report_delay_days", 2),
        ct, sv, pt, gn,
    ]])

    # RF probability
    fraud_prob = float(rf.predict_proba(features)[0][1])

    # Isolation Forest anomaly score (-1 = anomaly, 1 = normal)
    anomaly_label = int(iso.predict(features)[0])
    anomaly_score = float(iso.decision_function(features)[0])

    # Combined risk score (0-100)
    risk_score = round(fraud_prob * 70 + (1 if anomaly_label == -1 else 0) * 30, 1)
    risk_score = min(risk_score, 100.0)

    # Risk level
    if risk_score >= 70:
        risk_level = "High"
    elif risk_score >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # --- Explainability: top contributing factors ---
    feature_vals = dict(zip(FEATURE_COLS, features[0]))
    importances = dict(zip(FEATURE_COLS, rf.feature_importances_))

    explanations = []
    # Sort features by importance and flag concerning values
    for feat in sorted(importances, key=importances.get, reverse=True)[:5]:
        val = feature_vals[feat]
        imp = round(importances[feat] * 100, 1)
        reason = _explain_feature(feat, val)
        if reason:
            explanations.append({
                "feature": feat,
                "value": round(float(val), 2),
                "importance_pct": imp,
                "explanation": reason,
            })

    return {
        "fraud_probability": round(fraud_prob, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "anomaly_flag": anomaly_label == -1,
        "anomaly_score": round(anomaly_score, 4),
        "explanations": explanations,
    }


def _explain_feature(feat, val):
    """Generate human-readable explanation for a feature value."""
    explanations = {
        "claim_amount": lambda v: f"Claim amount ${v:,.0f} is {'unusually high' if v > 30000 else 'moderate' if v > 10000 else 'within normal range'}",
        "report_delay_days": lambda v: f"Report filed {int(v)} days after incident {'(significant delay - red flag)' if v > 10 else '(timely)' if v <= 3 else '(slight delay)'}",
        "num_witnesses": lambda v: f"{'No witnesses reported (increases suspicion)' if v == 0 else f'{int(v)} witness(es) present'}",
        "police_report_filed": lambda v: f"{'No police report filed (red flag)' if v == 0 else 'Police report on file'}",
        "credit_score": lambda v: f"Credit score {int(v)} is {'low (correlated with higher fraud risk)' if v < 500 else 'average' if v < 700 else 'good'}",
        "num_prior_claims": lambda v: f"{int(v)} prior claims {'(frequent claimant - elevated risk)' if v > 4 else '(normal history)'}",
        "has_violations": lambda v: f"{'Has prior violations (risk factor)' if v else 'No prior violations'}",
        "years_as_customer": lambda v: f"Customer for {int(v)} years {'(new customer - higher risk)' if v < 1 else '(established relationship)' if v > 5 else ''}",
        "annual_premium": lambda v: f"Annual premium ${v:,.0f}",
        "age": lambda v: f"Policyholder age {int(v)}",
    }
    fn = explanations.get(feat)
    return fn(val) if fn else None


# -------------------------------------------------------------------
if __name__ == "__main__":
    train_fraud_model()
