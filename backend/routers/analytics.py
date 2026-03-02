"""
Analytics API Router
=====================
Endpoints for dashboard statistics, distributions, and trend data.
"""

import sys
import os
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _load():
    claims = pd.read_csv(os.path.join(DATA_DIR, "claims.csv"))
    policies = pd.read_csv(os.path.join(DATA_DIR, "policies.csv"))
    claims["claim_amount"] = pd.to_numeric(claims["claim_amount"], errors="coerce")
    return claims, policies


@router.get("/summary")
def dashboard_summary():
    """Key metrics for the main dashboard."""
    try:
        claims, policies = _load()
        total_claims = len(claims)
        total_policies = len(policies)
        fraud_count = int(claims["is_fraud"].sum())
        fraud_rate = round(100 * fraud_count / total_claims, 1) if total_claims else 0
        avg_claim = round(float(claims["claim_amount"].mean()), 2)
        total_exposure = round(float(claims["claim_amount"].sum()), 2)
        avg_premium = round(float(policies["annual_premium"].mean()), 2)

        return {
            "status": "success",
            "data": {
                "total_claims": total_claims,
                "total_policies": total_policies,
                "fraud_count": fraud_count,
                "fraud_rate_pct": fraud_rate,
                "avg_claim_amount": avg_claim,
                "total_exposure": total_exposure,
                "avg_annual_premium": avg_premium,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/claims-by-type")
def claims_by_type():
    """Distribution of claims by type."""
    try:
        claims, _ = _load()
        dist = claims["claim_type"].astype(str).str.strip().str.title().value_counts().to_dict()
        return {"status": "success", "data": dist}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/claims-by-severity")
def claims_by_severity():
    """Distribution of claims by severity."""
    try:
        claims, _ = _load()
        dist = claims["severity"].astype(str).str.strip().str.title().value_counts().to_dict()
        return {"status": "success", "data": dist}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fraud-by-type")
def fraud_by_type():
    """Fraud count and rate broken down by claim type."""
    try:
        claims, _ = _load()
        claims["claim_type_clean"] = claims["claim_type"].astype(str).str.strip().str.title()
        grouped = claims.groupby("claim_type_clean").agg(
            total=("is_fraud", "count"),
            fraud=("is_fraud", "sum"),
        ).reset_index()
        grouped["fraud_rate"] = (grouped["fraud"] / grouped["total"] * 100).round(1)
        result = grouped.to_dict(orient="records")
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monthly-trends")
def monthly_trends():
    """Monthly claim counts and fraud counts over time."""
    try:
        claims, _ = _load()
        claims["claim_date_parsed"] = pd.to_datetime(claims["claim_date"], errors="coerce")
        claims.dropna(subset=["claim_date_parsed"], inplace=True)
        claims["month"] = claims["claim_date_parsed"].dt.to_period("M").astype(str)
        grouped = claims.groupby("month").agg(
            total_claims=("is_fraud", "count"),
            fraud_claims=("is_fraud", "sum"),
            total_amount=("claim_amount", "sum"),
        ).reset_index()
        grouped["total_amount"] = grouped["total_amount"].round(2)
        return {"status": "success", "data": grouped.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-distribution")
def risk_distribution():
    """Distribution of credit scores and age in the policyholder base."""
    try:
        _, policies = _load()
        age_bins = [18, 25, 35, 45, 55, 65, 75, 100]
        age_labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
        policies["age_group"] = pd.cut(policies["age"], bins=age_bins, labels=age_labels, right=False)
        age_dist = policies["age_group"].value_counts().sort_index().to_dict()
        # Convert Interval keys to strings
        age_dist = {str(k): int(v) for k, v in age_dist.items()}

        credit_bins = [300, 500, 600, 700, 800, 850]
        credit_labels = ["300-499", "500-599", "600-699", "700-799", "800-850"]
        policies["credit_group"] = pd.cut(policies["credit_score"], bins=credit_bins, labels=credit_labels, right=False)
        credit_dist = policies["credit_group"].value_counts().sort_index().to_dict()
        credit_dist = {str(k): int(v) for k, v in credit_dist.items()}

        state_dist = policies["state"].value_counts().head(10).to_dict()

        return {
            "status": "success",
            "data": {
                "age_distribution": age_dist,
                "credit_score_distribution": credit_dist,
                "top_states": state_dist,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/claim-amount-distribution")
def claim_amount_distribution():
    """Histogram data for claim amounts (fraud vs legitimate)."""
    try:
        claims, _ = _load()
        claims.dropna(subset=["claim_amount"], inplace=True)

        bins = [0, 1000, 5000, 10000, 25000, 50000, 100000, 200000, 500000]
        labels = ["0-1K", "1K-5K", "5K-10K", "10K-25K", "25K-50K", "50K-100K", "100K-200K", "200K+"]

        legit = claims[claims["is_fraud"] == 0]["claim_amount"]
        fraud = claims[claims["is_fraud"] == 1]["claim_amount"]

        legit_hist = pd.cut(legit, bins=bins, labels=labels, right=False).value_counts().sort_index()
        fraud_hist = pd.cut(fraud, bins=bins, labels=labels, right=False).value_counts().sort_index()

        return {
            "status": "success",
            "data": {
                "labels": labels,
                "legitimate": [int(x) for x in legit_hist.values],
                "fraudulent": [int(x) for x in fraud_hist.values],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
