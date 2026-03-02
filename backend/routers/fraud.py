"""
Fraud Detection API Router
============================
Endpoints for fraud scoring, batch analysis, and model metrics.
"""

import sys
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.fraud_detector import load_models, predict_fraud

router = APIRouter(prefix="/api/fraud", tags=["Fraud Detection"])

# Load models once at startup
rf_model, iso_model, encoders, evaluation = load_models()


# -------------------------------------------------------------------
# Request / Response schemas
# -------------------------------------------------------------------
class ClaimInput(BaseModel):
    """Input schema for a single claim fraud check."""
    age: int = Field(35, ge=18, le=100, description="Policyholder age")
    credit_score: int = Field(650, ge=300, le=850)
    annual_premium: float = Field(2000, ge=0)
    years_as_customer: int = Field(3, ge=0)
    num_prior_claims: int = Field(1, ge=0)
    has_violations: int = Field(0, ge=0, le=1)
    claim_amount: float = Field(5000, ge=0)
    num_witnesses: int = Field(1, ge=0)
    police_report_filed: int = Field(1, ge=0, le=1)
    report_delay_days: int = Field(2, ge=0)
    claim_type: str = Field("Auto", description="Auto, Property, Health, Life, Liability")
    severity: str = Field("Medium", description="Low, Medium, High")
    policy_type: str = Field("Standard", description="Basic, Standard, Premium, Comprehensive")
    gender: str = Field("Male", description="Male or Female")


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@router.post("/score")
def score_claim(claim: ClaimInput):
    """
    Score a single claim for fraud probability.
    Returns fraud probability, risk score, anomaly flag, and explanations.
    """
    try:
        result = predict_fraud(rf_model, iso_model, encoders, claim.model_dump())
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-metrics")
def get_model_metrics():
    """Return model evaluation metrics, feature importances, and confusion matrix."""
    return {
        "status": "success",
        "data": evaluation,
    }


@router.get("/feature-importances")
def get_feature_importances():
    """Return ranked feature importances from the trained model."""
    imp = evaluation.get("feature_importances", {})
    ranked = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    return {
        "status": "success",
        "data": [{"feature": k, "importance": v} for k, v in ranked],
    }
