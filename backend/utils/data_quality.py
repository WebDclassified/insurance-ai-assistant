"""
Data Quality & Standardization Module
=======================================
- Identify missing, inconsistent, or duplicate data
- Standardize data formats
- Generate data quality reports with actionable insights
- Clean and transform raw insurance data
"""

import os
import re
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_data():
    """Load claims and policies CSVs."""
    claims = pd.read_csv(os.path.join(DATA_DIR, "claims.csv"))
    policies = pd.read_csv(os.path.join(DATA_DIR, "policies.csv"))
    return claims, policies


# -------------------------------------------------------------------
# Missing value analysis
# -------------------------------------------------------------------
def check_missing_values(df: pd.DataFrame, name: str = "dataset") -> dict:
    """Identify missing values in each column."""
    total = len(df)
    issues = []
    summary = {}

    for col in df.columns:
        # Count NaN + empty strings + whitespace-only
        null_count = int(df[col].isna().sum())
        empty_count = int((df[col].astype(str).str.strip() == "").sum()) - null_count
        missing = null_count + max(empty_count, 0)

        pct = round(100 * missing / total, 2) if total > 0 else 0
        summary[col] = {"missing_count": missing, "missing_pct": pct}

        if missing > 0:
            severity = "critical" if pct > 10 else "warning" if pct > 2 else "info"
            issues.append({
                "column": col,
                "issue_type": "missing_values",
                "count": missing,
                "percentage": pct,
                "severity": severity,
                "recommendation": f"Fill or impute {missing} missing values in '{col}' ({pct}% of records).",
            })

    return {"dataset": name, "total_rows": total, "columns": summary, "issues": issues}


# -------------------------------------------------------------------
# Duplicate detection
# -------------------------------------------------------------------
def check_duplicates(df: pd.DataFrame, subset=None, name: str = "dataset") -> dict:
    """Find exact duplicate rows."""
    dups = df.duplicated(subset=subset, keep=False)
    dup_count = int(dups.sum())
    dup_groups = int(df[dups].groupby(list(df.columns if subset is None else subset)).ngroups) if dup_count > 0 else 0

    # Get example duplicate IDs
    examples = []
    if dup_count > 0 and "claim_id" in df.columns:
        dup_ids = df.loc[dups, "claim_id"].head(10).tolist()
        examples = dup_ids

    return {
        "dataset": name,
        "duplicate_rows": dup_count,
        "duplicate_groups": dup_groups,
        "severity": "critical" if dup_count > 10 else "warning" if dup_count > 0 else "ok",
        "example_ids": examples,
        "recommendation": f"Remove {dup_count} duplicate rows ({dup_groups} groups)." if dup_count > 0 else "No duplicates found.",
    }


# -------------------------------------------------------------------
# Consistency checks
# -------------------------------------------------------------------
def check_consistency(df: pd.DataFrame, name: str = "dataset") -> list:
    """Check for formatting inconsistencies across columns."""
    issues = []

    # 1. Inconsistent casing in categorical columns
    for col in ["claim_type", "severity", "policy_type", "gender", "marital_status"]:
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str)
        unique_raw = values.unique()
        unique_title = values.str.strip().str.title().unique()
        if len(unique_raw) > len(unique_title):
            bad = [v for v in unique_raw if v != v.strip().title()]
            issues.append({
                "column": col,
                "issue_type": "inconsistent_casing",
                "bad_values": bad[:10],
                "severity": "warning",
                "recommendation": f"Standardize casing in '{col}': found {len(bad)} variant(s).",
            })

    # 2. Invalid date formats
    for col in ["claim_date", "report_date", "policy_start_date"]:
        if col not in df.columns:
            continue
        invalid = []
        for i, val in df[col].dropna().items():
            try:
                datetime.strptime(str(val).strip(), "%Y-%m-%d")
            except ValueError:
                invalid.append(str(val))
        if invalid:
            issues.append({
                "column": col,
                "issue_type": "invalid_date_format",
                "bad_values": invalid[:10],
                "count": len(invalid),
                "severity": "critical",
                "recommendation": f"Fix {len(invalid)} invalid date(s) in '{col}'. Expected format: YYYY-MM-DD.",
            })

    # 3. Negative or unreasonable numeric values
    for col in ["claim_amount", "annual_premium", "credit_score", "age"]:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        negatives = int((numeric < 0).sum())
        if negatives > 0:
            issues.append({
                "column": col,
                "issue_type": "negative_values",
                "count": negatives,
                "severity": "critical",
                "recommendation": f"Investigate {negatives} negative value(s) in '{col}'.",
            })

    # 4. Outliers (values > 3 std deviations from mean)
    for col in ["claim_amount", "annual_premium"]:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(numeric) < 10:
            continue
        mean, std = numeric.mean(), numeric.std()
        outliers = int(((numeric - mean).abs() > 3 * std).sum())
        if outliers > 0:
            issues.append({
                "column": col,
                "issue_type": "statistical_outliers",
                "count": outliers,
                "threshold": f">{round(mean + 3*std, 2)}",
                "severity": "info",
                "recommendation": f"Review {outliers} statistical outlier(s) in '{col}' (>3 std dev).",
            })

    return issues


# -------------------------------------------------------------------
# Full data quality report
# -------------------------------------------------------------------
def generate_quality_report() -> dict:
    """Run all DQ checks and return a comprehensive report."""
    claims, policies = load_data()

    # Missing values
    claims_missing = check_missing_values(claims, "claims")
    policies_missing = check_missing_values(policies, "policies")

    # Duplicates
    claims_dups = check_duplicates(claims, name="claims")
    policies_dups = check_duplicates(policies, subset=["policy_id"], name="policies")

    # Consistency
    merged = claims.merge(policies, on="policy_id", how="left")
    consistency_issues = check_consistency(merged, "merged")

    # Aggregate all issues
    all_issues = (
        claims_missing["issues"]
        + policies_missing["issues"]
        + [claims_dups]
        + [policies_dups]
        + consistency_issues
    )

    # Compute overall health score (100 = perfect)
    critical_count = sum(1 for i in all_issues if isinstance(i, dict) and i.get("severity") == "critical")
    warning_count = sum(1 for i in all_issues if isinstance(i, dict) and i.get("severity") == "warning")
    health_score = max(0, 100 - critical_count * 15 - warning_count * 5)

    # Summary stats
    total_records = len(claims) + len(policies)
    total_fields = claims.shape[1] * len(claims) + policies.shape[1] * len(policies)

    return {
        "report_generated": datetime.now().isoformat(),
        "summary": {
            "total_records": total_records,
            "total_fields_checked": total_fields,
            "health_score": health_score,
            "health_grade": (
                "A" if health_score >= 90 else
                "B" if health_score >= 75 else
                "C" if health_score >= 60 else
                "D" if health_score >= 40 else "F"
            ),
            "critical_issues": critical_count,
            "warnings": warning_count,
            "total_issues": len(all_issues),
        },
        "missing_values": {
            "claims": claims_missing,
            "policies": policies_missing,
        },
        "duplicates": {
            "claims": claims_dups,
            "policies": policies_dups,
        },
        "consistency": consistency_issues,
    }


# -------------------------------------------------------------------
# Data cleaning / standardization
# -------------------------------------------------------------------
def clean_dataset() -> dict:
    """Apply automated cleaning and return before/after summary."""
    claims, policies = load_data()
    original_claims = len(claims)

    fixes = []

    # 1. Standardize casing
    for col in ["claim_type", "severity"]:
        if col in claims.columns:
            before = claims[col].nunique()
            claims[col] = claims[col].astype(str).str.strip().str.title()
            after = claims[col].nunique()
            if before != after:
                fixes.append(f"Standardized casing in '{col}': {before} -> {after} unique values")

    # 2. Fix invalid dates
    for col in ["claim_date", "report_date"]:
        if col in claims.columns:
            invalid_mask = claims[col].apply(lambda x: _is_invalid_date(x))
            n_fixed = int(invalid_mask.sum())
            if n_fixed > 0:
                claims.loc[invalid_mask, col] = pd.NaT
                fixes.append(f"Marked {n_fixed} invalid date(s) in '{col}' as NaT")

    # 3. Coerce claim_amount to numeric
    claims["claim_amount"] = pd.to_numeric(claims["claim_amount"], errors="coerce")
    missing_amount = int(claims["claim_amount"].isna().sum())
    if missing_amount > 0:
        median_amt = claims["claim_amount"].median()
        claims["claim_amount"].fillna(median_amt, inplace=True)
        fixes.append(f"Imputed {missing_amount} missing claim_amount(s) with median (${median_amt:,.2f})")

    # 4. Remove exact duplicates
    before_dedup = len(claims)
    claims.drop_duplicates(inplace=True)
    removed = before_dedup - len(claims)
    if removed > 0:
        fixes.append(f"Removed {removed} exact duplicate rows")

    # Save cleaned data
    claims.to_csv(os.path.join(DATA_DIR, "claims_cleaned.csv"), index=False)
    policies.to_csv(os.path.join(DATA_DIR, "policies_cleaned.csv"), index=False)

    return {
        "original_records": original_claims,
        "cleaned_records": len(claims),
        "fixes_applied": fixes,
        "output_files": ["claims_cleaned.csv", "policies_cleaned.csv"],
    }


def _is_invalid_date(val):
    try:
        datetime.strptime(str(val).strip(), "%Y-%m-%d")
        return False
    except (ValueError, TypeError):
        return True if pd.notna(val) and str(val).strip() != "" else False
