"""
Claims Processing API Router
==============================
Endpoints for claim classification, NLP extraction, settlement estimation, and priority routing.
"""

import sys
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.claims_processor import (
    classify_claim_type, classify_severity, extract_entities,
    estimate_settlement, determine_priority, process_claim,
)

router = APIRouter(prefix="/api/claims", tags=["Claims Processing"])


class ClaimTextInput(BaseModel):
    """Input for processing a claim from its description."""
    description: str = Field(..., min_length=10, description="Free-text claim description")
    claim_amount: float = Field(5000, ge=0, description="Claimed amount in USD")
    fraud_probability: float = Field(0.0, ge=0, le=1, description="Fraud score (0-1) if available")
    police_report: bool = Field(True, description="Whether a police report was filed")


@router.post("/process")
def process_claim_endpoint(data: ClaimTextInput):
    """
    Full claim processing pipeline:
    - Classify type and severity
    - Extract entities (NLP)
    - Estimate settlement
    - Determine priority routing
    """
    try:
        result = process_claim(
            description=data.description,
            claim_amount=data.claim_amount,
            fraud_probability=data.fraud_probability,
            police_report=data.police_report,
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify")
def classify_endpoint(data: ClaimTextInput):
    """Classify claim type from description text."""
    try:
        type_result = classify_claim_type(data.description)
        severity_result = classify_severity(data.description, data.claim_amount)
        return {
            "status": "success",
            "data": {"type": type_result, "severity": severity_result},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract")
def extract_endpoint(data: ClaimTextInput):
    """Extract entities and key information from claim description."""
    try:
        entities = extract_entities(data.description)
        return {"status": "success", "data": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
