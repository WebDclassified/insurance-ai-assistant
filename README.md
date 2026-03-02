# Insurance AI Assistant

An AI-powered insurance data assistant that demonstrates how data science and machine learning can address critical insurance industry pain points. Built with **Python/FastAPI** backend, **scikit-learn** ML models, and a **Chart.js** interactive frontend.

> **Domain Focus**: Insurance fraud detection — a $80B+/year problem where AI can increase detection from 10-20% (rule-based) to 88%+ (ML-based) while maintaining near-zero false positives.

---

## Features

### 1. Fraud Detection Engine
- **Random Forest classifier** trained on 1500+ claims to detect fraud (98% accuracy, 0.994 AUC-ROC)
- **Isolation Forest** anomaly detector for unsupervised fraud pattern detection
- **Explainability**: Top contributing factors shown for every prediction
- Real-time fraud risk scoring with probability, risk level, and anomaly flags

### 2. What-If Scenario Analysis *(Innovation)*
- Modify individual risk factors and instantly see how fraud risk changes
- 12 pre-built scenarios (delay changes, credit shifts, witness variations, etc.)
- Helps claims adjusters understand *which factors drive risk* for each claim
- Supports regulatory explainability requirements (GDPR, AI Act)

### 3. Risk Profile Comparison *(Innovation)*
- Side-by-side statistical profiles of fraudulent vs legitimate claims
- Key differentiator analysis with percentage differences
- Interactive bar chart comparing fraud vs legitimate averages
- Insights like "Fraudulent claims average 22 days report delay vs 5 days for legitimate"

### 4. Batch Fraud Scoring *(Innovation)*
- Score multiple claims simultaneously via REST API
- Returns per-claim results plus aggregate summary (high/medium/low risk counts)
- Enables portfolio-level fraud risk assessment

### 5. Claims Processing Intelligence
- **Auto-categorization** of claims by type (Collision, Theft, Weather, Medical, Liability, Property, Fire) using keyword-based NLP
- **Severity classification** combining text analysis with claim amount heuristics
- **Entity extraction** from unstructured claim descriptions (monetary amounts, dates, locations, injuries)
- **Settlement estimation** with confidence ranges
- **Priority routing** to appropriate claims handler based on severity, fraud risk, and amount

### 6. Data Quality & Standardization
- Detects **missing values**, **duplicate records**, **inconsistent formatting**, **invalid dates**, and **statistical outliers**
- Generates comprehensive data health reports with an overall health score (A-F grading)
- Automated cleaning: casing normalization, date validation, duplicate removal, numeric imputation
- Actionable recommendations for each issue found

### 7. Predictive Analytics Dashboard
- Interactive charts: claims by type, severity distribution, fraud rates, monthly trends
- Policyholder demographics: age, credit score, and geographic distributions
- Claim amount histograms (fraud vs. legitimate)
- Model performance metrics with confusion matrix display

### 8. Insurance Domain Knowledge Center
- Industry glossary with 12 key insurance terms (premium, underwriting, loss ratio, subrogation, etc.)
- Regulatory context: GDPR, HIPAA, NAIC, EU AI Act compliance considerations
- Innovation highlights and business impact framing

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **ML/AI** | scikit-learn (Random Forest, Isolation Forest), pandas, NumPy |
| **Frontend** | Vanilla JavaScript, Chart.js 4.x, CSS3 |
| **Data** | Synthetic dataset (1500 policies, 1550+ claims, ~16% fraud rate) |

---

## Project Structure

```
insurance_ai_assistant/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   ├── generate_dataset.py            # Synthetic data generator
│   ├── policies.csv                   # 1500 policyholder records
│   └── claims.csv                     # 1550+ claim records
├── models/
│   ├── fraud_detector.py              # RF + Isolation Forest training & inference
│   ├── claims_processor.py            # NLP classification, entity extraction, settlement
│   └── trained/                       # Serialized models & evaluation metrics
│       ├── fraud_rf.joblib
│       ├── fraud_iso.joblib
│       ├── label_encoders.joblib
│       └── evaluation.json
├── backend/
│   ├── main.py                        # FastAPI application entry point
│   ├── routers/
│   │   ├── fraud.py                   # POST /api/fraud/score, GET /api/fraud/model-metrics
│   │   ├── claims.py                  # POST /api/claims/process, /classify, /extract
│   │   ├── data_quality.py            # GET /api/data-quality/report, POST /clean
│   │   └── analytics.py              # GET /api/analytics/summary, trends, distributions
│   └── utils/
│       └── data_quality.py            # DQ checks: missing, dupes, consistency, outliers
└── frontend/
    ├── index.html                     # Single-page application
    ├── css/styles.css                 # Responsive stylesheet
    └── js/app.js                      # Navigation, API calls, Chart.js rendering
```

---

## Setup & Run

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone or navigate to the project
cd insurance_ai_assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic data (if not already present)
python data/generate_dataset.py

# 4. Train ML models
python models/fraud_detector.py

# 5. Start the server
cd backend
uvicorn main:app --reload --port 8000
```

### Access
- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (auto-generated Swagger UI)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/fraud/score` | Score a single claim for fraud risk |
| `POST` | `/api/fraud/batch-score` | Score multiple claims at once |
| `POST` | `/api/fraud/what-if` | What-if scenario analysis for a claim |
| `GET`  | `/api/fraud/risk-profile` | Fraud vs legitimate risk profile comparison |
| `GET`  | `/api/fraud/model-metrics` | Model evaluation metrics |
| `GET`  | `/api/fraud/feature-importances` | Ranked feature importances |
| `POST` | `/api/claims/process` | Full claim processing pipeline |
| `POST` | `/api/claims/classify` | Classify claim type & severity |
| `POST` | `/api/claims/extract` | Extract entities from description |
| `GET`  | `/api/data-quality/report` | Full data quality report |
| `POST` | `/api/data-quality/clean` | Clean & standardize data |
| `GET`  | `/api/analytics/summary` | Dashboard summary metrics |
| `GET`  | `/api/analytics/claims-by-type` | Claims distribution by type |
| `GET`  | `/api/analytics/claims-by-severity` | Claims distribution by severity |
| `GET`  | `/api/analytics/fraud-by-type` | Fraud rate by claim type |
| `GET`  | `/api/analytics/monthly-trends` | Monthly claim/fraud trends |
| `GET`  | `/api/analytics/risk-distribution` | Policyholder demographics |
| `GET`  | `/api/analytics/claim-amount-distribution` | Claim amounts histogram |

---

## Model Evaluation

**Fraud Detection (Random Forest)**
- **Accuracy**: 98.2%
- **Precision**: 100% (no false positives on test set)
- **Recall**: 88.1% (catches 88% of actual fraud)
- **F1 Score**: 93.7%
- **ROC-AUC**: 0.994

Top predictive features: `report_delay_days`, `num_witnesses`, `claim_amount`, `credit_score`, `annual_premium`

---

## Business Impact

| Metric | Before (Rule-Based) | After (AI-Powered) | Improvement |
|--------|---------------------|--------------------|-------------|
| Fraud Detection Rate | 10-20% | 88.1% | **4-8x increase** |
| False Positive Rate | 30-50% | 0% | **Eliminated** |
| Claims Processing Time | 15-30 days | Real-time | **~99% reduction** |
| Data Quality Score | Grade C (72) | Grade A (95+) | **+32% improvement** |

**Estimated ROI**: For an insurer processing 100,000 claims/year with 15.8% fraud rate and $15,000 avg fraud cost, this system could save **~$161 million/year** in additional fraud detection.

---

## Innovation Highlights

1. **Dual-Model Ensemble** — Supervised (Random Forest) + Unsupervised (Isolation Forest) with weighted 70/30 scoring catches both known and novel fraud patterns
2. **What-If Scenario Analysis** — Explainable AI that shows how each factor influences fraud risk, meeting regulatory requirements for decision transparency
3. **Risk Profile Engine** — Automated statistical comparison of fraud vs legitimate claim characteristics with key differentiator insights
4. **Batch Scoring API** — Portfolio-level fraud assessment for enterprise workflows
5. **Auto Data Quality** — One-click dataset cleaning with health scoring (A-F grading system)
6. **Full-Stack Integration** — End-to-end from synthetic data generation to trained ML models to REST API to interactive web dashboard

---

## License

This project is for educational and demonstration purposes.
