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


@router.post("/batch-score")
def batch_score(claims: list[ClaimInput]):
    """
    Score multiple claims at once. Returns a list of risk assessments.
    Useful for bulk analysis of historical or incoming claims.
    """
    try:
        results = []
        for claim in claims:
            result = predict_fraud(rf_model, iso_model, encoders, claim.model_dump())
            results.append(result)
        summary = {
            "total": len(results),
            "high_risk": sum(1 for r in results if r["risk_level"] == "High"),
            "medium_risk": sum(1 for r in results if r["risk_level"] == "Medium"),
            "low_risk": sum(1 for r in results if r["risk_level"] == "Low"),
            "avg_risk_score": round(sum(r["risk_score"] for r in results) / len(results), 1) if results else 0,
        }
        return {"status": "success", "data": {"results": results, "summary": summary}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/what-if")
def what_if_analysis(claim: ClaimInput):
    """
    What-If Scenario Analysis: Shows how changing individual features
    would affect the fraud risk score. Helps adjusters understand
    which factors are driving the risk.
    """
    try:
        base = claim.model_dump()
        base_result = predict_fraud(rf_model, iso_model, encoders, base)
        base_score = base_result["risk_score"]

        scenarios = []
        # Test various parameter changes
        variations = [
            ("report_delay_days", "Delay increased to 30 days", {"report_delay_days": 30}),
            ("report_delay_days", "Delay reduced to 0 days", {"report_delay_days": 0}),
            ("num_prior_claims", "Prior claims increased to 5", {"num_prior_claims": 5}),
            ("num_prior_claims", "No prior claims", {"num_prior_claims": 0}),
            ("credit_score", "Credit score dropped to 400", {"credit_score": 400}),
            ("credit_score", "Credit score at 800", {"credit_score": 800}),
            ("claim_amount", "Claim amount doubled", {"claim_amount": base["claim_amount"] * 2}),
            ("claim_amount", "Claim amount halved", {"claim_amount": base["claim_amount"] / 2}),
            ("police_report_filed", "No police report", {"police_report_filed": 0}),
            ("police_report_filed", "Police report filed", {"police_report_filed": 1}),
            ("num_witnesses", "No witnesses", {"num_witnesses": 0}),
            ("num_witnesses", "3 witnesses present", {"num_witnesses": 3}),
        ]
        for feature, label, changes in variations:
            modified = {**base, **changes}
            mod_result = predict_fraud(rf_model, iso_model, encoders, modified)
            delta = mod_result["risk_score"] - base_score
            scenarios.append({
                "feature": feature,
                "scenario": label,
                "new_score": mod_result["risk_score"],
                "delta": round(delta, 1),
                "new_level": mod_result["risk_level"],
                "direction": "increase" if delta > 0 else "decrease" if delta < 0 else "no_change",
            })
        return {
            "status": "success",
            "data": {
                "base_score": base_score,
                "base_level": base_result["risk_level"],
                "scenarios": scenarios,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-profile")
def risk_profile():
    """
    Generate risk profiles from the dataset - shows distribution of risk
    factors across the policyholder base. Useful for portfolio analysis.
    """
    try:
        import pandas as pd
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        claims = pd.read_csv(os.path.join(base_dir, "data", "claims.csv"))
        policies = pd.read_csv(os.path.join(base_dir, "data", "policies.csv"))
        merged = claims.merge(policies, on="policy_id", how="left")

        # Risk segment analysis
        fraud = merged[merged["is_fraud"] == 1]
        legit = merged[merged["is_fraud"] == 0]

        def safe_mean(series):
            val = series.mean()
            return round(float(val), 2) if pd.notna(val) else 0

        profile = {
            "fraud_profile": {
                "avg_age": safe_mean(fraud["age"]),
                "avg_credit_score": safe_mean(fraud["credit_score"]),
                "avg_claim_amount": safe_mean(pd.to_numeric(fraud["claim_amount"], errors="coerce")),
                "avg_prior_claims": safe_mean(fraud["num_prior_claims"]),
                "avg_report_delay": safe_mean(fraud["report_delay_days"]),
                "pct_with_violations": round(float(fraud["has_violations"].mean() * 100), 1) if len(fraud) else 0,
                "pct_no_police_report": round(float((1 - fraud["police_report_filed"].mean()) * 100), 1) if len(fraud) else 0,
                "count": len(fraud),
            },
            "legitimate_profile": {
                "avg_age": safe_mean(legit["age"]),
                "avg_credit_score": safe_mean(legit["credit_score"]),
                "avg_claim_amount": safe_mean(pd.to_numeric(legit["claim_amount"], errors="coerce")),
                "avg_prior_claims": safe_mean(legit["num_prior_claims"]),
                "avg_report_delay": safe_mean(legit["report_delay_days"]),
                "pct_with_violations": round(float(legit["has_violations"].mean() * 100), 1) if len(legit) else 0,
                "pct_no_police_report": round(float((1 - legit["police_report_filed"].mean()) * 100), 1) if len(legit) else 0,
                "count": len(legit),
            },
            "key_differentiators": [],
        }
        # Identify biggest differences
        fp = profile["fraud_profile"]
        lp = profile["legitimate_profile"]
        diffs = [
            ("Report Delay", fp["avg_report_delay"], lp["avg_report_delay"], "days"),
            ("Prior Claims", fp["avg_prior_claims"], lp["avg_prior_claims"], "claims"),
            ("Credit Score", fp["avg_credit_score"], lp["avg_credit_score"], "score"),
            ("Claim Amount", fp["avg_claim_amount"], lp["avg_claim_amount"], "$"),
            ("Violation Rate", fp["pct_with_violations"], lp["pct_with_violations"], "%"),
        ]
        for name, fraud_val, legit_val, unit in diffs:
            if legit_val != 0:
                pct_diff = round(abs(fraud_val - legit_val) / abs(legit_val) * 100, 1)
            else:
                pct_diff = 0
            profile["key_differentiators"].append({
                "factor": name,
                "fraud_avg": fraud_val,
                "legit_avg": legit_val,
                "difference_pct": pct_diff,
                "unit": unit,
                "insight": f"Fraudulent claims average {fraud_val}{unit} vs {legit_val}{unit} for legitimate claims."
            })
        return {"status": "success", "data": profile}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
