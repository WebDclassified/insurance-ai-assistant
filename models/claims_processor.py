"""
Claims Processing Intelligence Module
=======================================
- Auto-categorize claims by type and severity
- Extract key information from claim descriptions (NLP)
- Estimate claim settlement amounts
- Priority routing based on urgency
"""

import re
import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# -------------------------------------------------------------------
# Keyword-based NLP extraction rules
# -------------------------------------------------------------------
TYPE_KEYWORDS = {
    "Auto": ["vehicle", "car", "collision", "accident", "bumper", "fender",
             "rear-ended", "t-boned", "highway", "traffic", "driver",
             "hydroplaned", "deer", "windshield", "airbag", "pileup",
             "stolen", "whiplash", "driveway", "driving"],
    "Property": ["roof", "pipe", "flood", "fire", "kitchen", "hail",
                 "shingle", "basement", "tree", "window", "smoke",
                 "wildfire", "siding", "water damage", "theft", "jewelry",
                 "art collection", "electronics"],
    "Health": ["surgery", "hospital", "physical therapy", "emergency room",
               "fracture", "hernia", "medications", "pharmacy", "ACL",
               "appendectomy", "medical", "billing", "sessions", "visits"],
    "Life": ["death", "deceased", "beneficiary", "life insurance",
             "mortality", "funeral"],
    "Liability": ["liability", "lawsuit", "negligence", "third party",
                  "premises", "injured party"],
}

SEVERITY_KEYWORDS = {
    "High": ["total loss", "fire", "death", "severe", "structural",
             "hospitalized", "surgery", "pileup", "multiple", "stolen",
             "destroyed", "$50,000", "extensive"],
    "Medium": ["moderate", "cracked", "dented", "therapy", "outpatient",
               "hernia", "scratches", "damaged", "broken"],
    "Low": ["minor", "cosmetic", "small", "scratches", "routine", "parked"],
}

URGENCY_RULES = {
    "Critical": {"severity": "High", "fraud_prob_above": 0.7},
    "High": {"severity": "High"},
    "Medium": {"severity": "Medium"},
    "Low": {"severity": "Low"},
}

# Entity extraction patterns
PATTERNS = {
    "monetary_amounts": r"\$[\d,]+(?:\.\d{2})?",
    "dates": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    "report_numbers": r"#\d{4,}",
    "locations": r"(?:Highway|Route|I-|SR-|US-)\s*\S+|(?:Main St|Park Ave|Oak Rd)",
    "injury_mentions": r"\b(?:whiplash|fracture|injury|injuries|injured|back pain|hernia|ACL)\b",
    "vehicle_mentions": r"\b(?:vehicle|car|truck|SUV|sedan)\b",
}


def classify_claim_type(description: str) -> dict:
    """Classify claim type from description text using keyword matching."""
    desc_lower = description.lower()
    scores = {}
    for ctype, keywords in TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in desc_lower)
        if score > 0:
            scores[ctype] = score

    if not scores:
        return {"predicted_type": "Unknown", "confidence": 0.0, "scores": {}}

    total = sum(scores.values())
    best = max(scores, key=scores.get)
    confidence = round(scores[best] / total, 3) if total > 0 else 0.0

    return {
        "predicted_type": best,
        "confidence": confidence,
        "scores": {k: round(v / total, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])},
    }


def classify_severity(description: str, claim_amount: float = 0) -> dict:
    """Classify severity from description text and claim amount."""
    desc_lower = description.lower()
    scores = {}
    for sev, keywords in SEVERITY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in desc_lower)
        scores[sev] = score

    # Boost based on claim amount
    if claim_amount > 30000:
        scores["High"] = scores.get("High", 0) + 3
    elif claim_amount > 10000:
        scores["Medium"] = scores.get("Medium", 0) + 2
    else:
        scores["Low"] = scores.get("Low", 0) + 1

    total = sum(scores.values())
    best = max(scores, key=scores.get) if total > 0 else "Medium"
    confidence = round(scores[best] / total, 3) if total > 0 else 0.33

    return {
        "predicted_severity": best,
        "confidence": confidence,
    }


def extract_entities(description: str) -> dict:
    """Extract structured entities from unstructured claim description."""
    entities = {}
    for name, pattern in PATTERNS.items():
        matches = re.findall(pattern, description, re.IGNORECASE)
        entities[name] = matches if matches else []

    # Determine key facts
    has_injury = len(entities.get("injury_mentions", [])) > 0
    has_vehicle = len(entities.get("vehicle_mentions", [])) > 0
    has_police = bool(re.search(r"police|report|officer", description, re.IGNORECASE))
    has_witnesses = bool(re.search(r"witness", description, re.IGNORECASE))

    entities["key_flags"] = {
        "injury_reported": has_injury,
        "vehicle_involved": has_vehicle,
        "police_involvement": has_police,
        "witnesses_mentioned": has_witnesses,
    }
    return entities


def estimate_settlement(claim_type: str, severity: str, claim_amount: float,
                        has_injury: bool = False, police_report: bool = True) -> dict:
    """
    Estimate likely settlement amount based on claim characteristics.
    Uses industry-average multipliers.
    """
    # Base multipliers by type
    type_multiplier = {
        "Auto": 0.75, "Property": 0.80, "Health": 0.85,
        "Life": 0.95, "Liability": 0.70, "Unknown": 0.70,
    }.get(claim_type, 0.75)

    # Severity adjustment
    severity_adj = {"Low": 0.90, "Medium": 1.0, "High": 1.10}.get(severity, 1.0)

    # Additional adjustments
    injury_adj = 1.15 if has_injury else 1.0
    police_adj = 1.05 if police_report else 0.95

    estimated = claim_amount * type_multiplier * severity_adj * injury_adj * police_adj
    low_est = estimated * 0.85
    high_est = estimated * 1.15

    return {
        "estimated_settlement": round(estimated, 2),
        "range_low": round(low_est, 2),
        "range_high": round(high_est, 2),
        "factors": {
            "type_multiplier": type_multiplier,
            "severity_adjustment": severity_adj,
            "injury_adjustment": injury_adj,
            "police_report_adjustment": police_adj,
        },
    }


def determine_priority(severity: str, fraud_probability: float = 0.0,
                       claim_amount: float = 0.0) -> dict:
    """Determine routing priority for a claim."""
    score = 0

    # Severity component (0-40)
    score += {"High": 40, "Medium": 20, "Low": 5}.get(severity, 10)

    # Fraud risk component (0-35)
    score += int(fraud_probability * 35)

    # Amount component (0-25)
    if claim_amount > 50000:
        score += 25
    elif claim_amount > 20000:
        score += 15
    elif claim_amount > 5000:
        score += 8

    # Map to priority level
    if score >= 70:
        priority = "Critical"
        routing = "Senior Claims Manager + SIU Review"
    elif score >= 45:
        priority = "High"
        routing = "Senior Claims Adjuster"
    elif score >= 25:
        priority = "Medium"
        routing = "Claims Adjuster"
    else:
        priority = "Low"
        routing = "Auto-Processing Queue"

    return {
        "priority": priority,
        "priority_score": min(score, 100),
        "routing": routing,
    }


def process_claim(description: str, claim_amount: float = 0.0,
                  fraud_probability: float = 0.0,
                  police_report: bool = True) -> dict:
    """Full claim processing pipeline: classify, extract, estimate, route."""
    type_result = classify_claim_type(description)
    severity_result = classify_severity(description, claim_amount)
    entities = extract_entities(description)
    has_injury = entities["key_flags"]["injury_reported"]

    settlement = estimate_settlement(
        type_result["predicted_type"],
        severity_result["predicted_severity"],
        claim_amount,
        has_injury,
        police_report,
    )

    priority = determine_priority(
        severity_result["predicted_severity"],
        fraud_probability,
        claim_amount,
    )

    return {
        "classification": type_result,
        "severity": severity_result,
        "entities": entities,
        "settlement_estimate": settlement,
        "priority_routing": priority,
    }
