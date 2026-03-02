"""
Data Quality API Router
========================
Endpoints for data quality reports, cleaning, and standardization.
"""

import sys
import os
from fastapi import APIRouter, HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_quality import generate_quality_report, clean_dataset

router = APIRouter(prefix="/api/data-quality", tags=["Data Quality"])


@router.get("/report")
def get_quality_report():
    """
    Run all data quality checks and return comprehensive report.
    Includes: missing values, duplicates, consistency, health score.
    """
    try:
        report = generate_quality_report()
        return {"status": "success", "data": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clean")
def run_cleaning():
    """
    Apply automated cleaning rules to the dataset:
    - Standardize casing
    - Fix invalid dates
    - Impute missing numeric values
    - Remove duplicates
    Returns summary of changes made.
    """
    try:
        result = clean_dataset()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
