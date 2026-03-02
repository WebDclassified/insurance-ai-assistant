"""
Synthetic Insurance Dataset Generator
======================================
Generates 1500+ realistic insurance records including:
- Structured: policy details, claim amounts, policyholder demographics
- Unstructured: free-text claim descriptions
- Intentional data quality issues for the DQ module to detect

Output: claims.csv, policies.csv
"""

import csv
import random
import os
from datetime import datetime, timedelta

random.seed(42)

# -------------------------------------------------------------------
# Constants & lookup tables
# -------------------------------------------------------------------
CLAIM_TYPES = ["Auto", "Property", "Health", "Life", "Liability"]
SEVERITY_MAP = {"Low": (100, 2000), "Medium": (2000, 15000), "High": (15000, 80000)}
STATES = [
    "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
]
OCCUPATIONS = [
    "Software Engineer", "Teacher", "Nurse", "Mechanic", "Accountant",
    "Sales Manager", "Electrician", "Lawyer", "Driver", "Chef",
    "Retail Worker", "Construction Worker", "Doctor", "Freelancer", "Student",
]
GENDERS = ["Male", "Female"]
MARITAL = ["Single", "Married", "Divorced", "Widowed"]
POLICY_TYPES = ["Basic", "Standard", "Premium", "Comprehensive"]

# Fraud patterns (for description generation)
AUTO_LEGIT = [
    "Rear-ended at traffic light on {road}. Minor bumper damage. Police report filed #{report}.",
    "Side collision at intersection of {road} and Main St. Driver uninjured. Damage to passenger door.",
    "Hit a deer on {road} at night. Windshield cracked and front hood dented.",
    "Parked vehicle struck in grocery store lot. Scratches on driver side panel.",
    "Hydroplaned on wet road during rainstorm. Slid into guardrail. Airbags did not deploy.",
    "T-boned by another vehicle running a red light. Moderate front-end damage.",
    "Backing out of driveway, struck neighbor's fence. Minor cosmetic damage to trunk.",
    "Multi-vehicle pileup on highway {road}. Moderate rear and side damage.",
]
AUTO_FRAUD = [
    "Total loss claimed on {road}. Vehicle allegedly caught fire overnight. No witnesses.",
    "Claiming severe whiplash from minor fender bender at {road}. Seven medical visits in two weeks.",
    "Vehicle reported stolen from driveway. Found burned 50 miles away. High-value aftermarket parts claimed.",
    "Three passengers all reporting identical back injuries from low-speed collision at {road}.",
    "Claimed vehicle was brand new condition; DMV records indicate salvage title.",
    "Accident reported two weeks after alleged occurrence. No police report. Workshop estimate inflated.",
]
PROPERTY_LEGIT = [
    "Pipe burst in basement during winter freeze. Water damage to flooring and drywall.",
    "Tree fell on roof during storm on {date_str}. Structural damage to attic and bedroom.",
    "Kitchen fire caused by unattended stove. Damage limited to kitchen area. Fire dept responded.",
    "Hail storm damaged roof shingles and broke two windows. Neighbors also filing claims.",
    "Smoke damage from nearby wildfire. Exterior siding discolored, HVAC system clogged.",
]
PROPERTY_FRAUD = [
    "Entire inventory of home electronics claimed destroyed by small kitchen fire. Receipts are photocopies.",
    "Claimed rare art collection worth $50,000 destroyed in basement flood. No prior insurance rider for valuables.",
    "Reporting theft of jewelry during vacation. No forced entry, alarm was disabled.",
]
HEALTH_LEGIT = [
    "Emergency room visit for fractured wrist from fall. X-ray and cast applied.",
    "Routine surgery for appendectomy. Three-day hospital stay. Full recovery expected.",
    "Physical therapy sessions for knee rehabilitation following ACL surgery.",
    "Outpatient procedure for hernia repair. Discharged same day.",
]
HEALTH_FRAUD = [
    "Billing for 15 physical therapy sessions. Patient records show only 6 visits.",
    "Claiming expensive brand-name medications when pharmacy records show generic substitutes dispensed.",
    "Multiple claims from different providers for the same date of service.",
]
GENERIC_LEGIT = [
    "Standard claim for covered incident. Documentation complete and verified.",
    "Incident occurred at insured premises. Damage consistent with reported circumstances.",
]
GENERIC_FRAUD = [
    "High-value claim filed one week after policy inception. No prior claim history.",
    "Claimant has filed similar claims with three different insurers in the past 12 months.",
]

ROADS = ["Highway 101", "I-95", "Route 66", "SR-520", "US-1", "I-10", "Park Ave", "Oak Rd"]


def random_date(start_year=2022, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def generate_description(claim_type, is_fraud):
    """Pick a template and fill in placeholders."""
    road = random.choice(ROADS)
    report = random.randint(100000, 999999)
    date_str = random_date().strftime("%B %d, %Y")

    if claim_type == "Auto":
        pool = AUTO_FRAUD if is_fraud else AUTO_LEGIT
    elif claim_type == "Property":
        pool = PROPERTY_FRAUD if is_fraud else PROPERTY_LEGIT
    elif claim_type == "Health":
        pool = HEALTH_FRAUD if is_fraud else HEALTH_LEGIT
    else:
        pool = GENERIC_FRAUD if is_fraud else GENERIC_LEGIT

    template = random.choice(pool)
    return template.format(road=road, report=report, date_str=date_str)


def generate_policies(n=1500):
    """Generate policyholder records."""
    policies = []
    for i in range(1, n + 1):
        pid = f"POL-{i:05d}"
        age = random.randint(18, 75)
        gender = random.choice(GENDERS)
        state = random.choice(STATES)
        occupation = random.choice(OCCUPATIONS)
        marital = random.choice(MARITAL)
        credit_score = random.randint(300, 850)
        policy_type = random.choice(POLICY_TYPES)
        annual_premium = round(random.uniform(500, 8000), 2)
        policy_start = random_date(2020, 2024)
        years_as_customer = random.randint(0, 15)
        num_prior_claims = random.randint(0, 8)
        has_violations = random.choice([0, 0, 0, 1])  # 25% have violations

        policies.append({
            "policy_id": pid,
            "age": age,
            "gender": gender,
            "state": state,
            "occupation": occupation,
            "marital_status": marital,
            "credit_score": credit_score,
            "policy_type": policy_type,
            "annual_premium": annual_premium,
            "policy_start_date": policy_start.strftime("%Y-%m-%d"),
            "years_as_customer": years_as_customer,
            "num_prior_claims": num_prior_claims,
            "has_violations": has_violations,
        })
    return policies


def generate_claims(policies, avg_claims_per_policy=1.2):
    """Generate claim records linked to policies. ~12% fraud rate."""
    claims = []
    claim_id = 1

    for pol in policies:
        n_claims = max(0, int(random.gauss(avg_claims_per_policy, 0.8)))
        if n_claims == 0 and random.random() < 0.6:
            n_claims = 1  # ensure most policies have at least one claim

        for _ in range(n_claims):
            # Fraud probability influenced by risk factors
            fraud_base = 0.10
            if pol["num_prior_claims"] > 4:
                fraud_base += 0.08
            if pol["credit_score"] < 500:
                fraud_base += 0.06
            if pol["has_violations"]:
                fraud_base += 0.04
            if pol["years_as_customer"] < 1:
                fraud_base += 0.05

            is_fraud = 1 if random.random() < fraud_base else 0

            claim_type = random.choice(CLAIM_TYPES)
            severity = random.choices(
                ["Low", "Medium", "High"],
                weights=[0.5, 0.35, 0.15] if not is_fraud else [0.2, 0.3, 0.5],
            )[0]

            lo, hi = SEVERITY_MAP[severity]
            claim_amount = round(random.uniform(lo, hi), 2)
            if is_fraud:
                claim_amount = round(claim_amount * random.uniform(1.3, 2.5), 2)

            claim_date = random_date(2022, 2025)
            report_delay = random.randint(0, 5) if not is_fraud else random.randint(2, 30)
            report_date = claim_date + timedelta(days=report_delay)

            witnesses = random.randint(0, 4) if not is_fraud else random.choices([0, 1], weights=[0.7, 0.3])[0]
            police_report = random.choice([0, 1]) if not is_fraud else random.choices([0, 1], weights=[0.6, 0.4])[0]

            description = generate_description(claim_type, is_fraud)

            claim = {
                "claim_id": f"CLM-{claim_id:06d}",
                "policy_id": pol["policy_id"],
                "claim_date": claim_date.strftime("%Y-%m-%d"),
                "report_date": report_date.strftime("%Y-%m-%d"),
                "claim_type": claim_type,
                "severity": severity,
                "claim_amount": claim_amount,
                "description": description,
                "num_witnesses": witnesses,
                "police_report_filed": police_report,
                "report_delay_days": report_delay,
                "is_fraud": is_fraud,
            }

            # --- Inject data quality issues into ~5% of records ---
            if random.random() < 0.02:
                claim["claim_amount"] = ""  # missing value
            if random.random() < 0.02:
                claim["claim_type"] = claim_type.lower()  # inconsistent casing
            if random.random() < 0.01:
                claim["claim_date"] = "13/32/2023"  # invalid date
            if random.random() < 0.01:
                claim["severity"] = "MEDIUM"  # inconsistent casing
            if random.random() < 0.015:
                claim["state_override"] = "California"  # full name instead of abbrev

            claims.append(claim)
            claim_id += 1

    # Inject ~20 exact duplicate rows for DQ detection
    for _ in range(20):
        dup = random.choice(claims).copy()
        claims.append(dup)

    random.shuffle(claims)
    return claims


def save_csv(rows, path, fieldnames=None):
    if not fieldnames:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {len(rows)} rows -> {path}")


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("Generating policyholder data...")
    policies = generate_policies(1500)
    save_csv(policies, os.path.join(out_dir, "policies.csv"))

    print("Generating claims data...")
    claims = generate_claims(policies)
    save_csv(
        claims,
        os.path.join(out_dir, "claims.csv"),
        fieldnames=[
            "claim_id", "policy_id", "claim_date", "report_date",
            "claim_type", "severity", "claim_amount", "description",
            "num_witnesses", "police_report_filed", "report_delay_days",
            "is_fraud",
        ],
    )

    # Print summary
    fraud_count = sum(1 for c in claims if c["is_fraud"] == 1)
    print(f"\nDataset Summary:")
    print(f"  Policies: {len(policies)}")
    print(f"  Claims:   {len(claims)}")
    print(f"  Fraudulent claims: {fraud_count} ({100*fraud_count/len(claims):.1f}%)")


if __name__ == "__main__":
    main()
