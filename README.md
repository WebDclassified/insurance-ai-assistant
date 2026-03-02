# Insurance AI Assistant

An AI-powered insurance data assistant that demonstrates how data science and machine learning can address critical insurance industry pain points. Built with **Python/FastAPI** backend, **scikit-learn** ML models, and a **Chart.js** interactive frontend.

---

## Features

### 1. Fraud Detection Engine
- **Random Forest classifier** trained on 1500+ claims to detect fraud (98% accuracy, 0.994 AUC-ROC)
- **Isolation Forest** anomaly detector for unsupervised fraud pattern detection
- **Explainability**: Top contributing factors shown for every prediction
- Real-time fraud risk scoring with probability, risk level, and anomaly flags

### 2. Claims Processing Intelligence
- **Auto-categorization** of claims by type (Auto, Property, Health, Life, Liability) using keyword-based NLP
- **Severity classification** combining text analysis with claim amount heuristics
- **Entity extraction** from unstructured claim descriptions (monetary amounts, dates, locations, injuries)
- **Settlement estimation** with confidence ranges
- **Priority routing** to appropriate claims handler based on severity, fraud risk, and amount

### 3. Data Quality & Standardization
- Detects **missing values**, **duplicate records**, **inconsistent formatting**, **invalid dates**, and **statistical outliers**
- Generates comprehensive data health reports with an overall health score (A-F grading)
- Automated cleaning: casing normalization, date validation, duplicate removal, numeric imputation
- Actionable recommendations for each issue found

### 4. Predictive Analytics Dashboard
- Interactive charts: claims by type, severity distribution, fraud rates, monthly trends
- Policyholder demographics: age, credit score, and geographic distributions
- Claim amount histograms (fraud vs. legitimate)
- Model performance metrics with confusion matrix display

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
| `POST` | `/api/fraud/score` | Score a claim for fraud risk |
| `GET`  | `/api/fraud/model-metrics` | Model evaluation metrics |
| `GET`  | `/api/fraud/feature-importances` | Ranked feature importances |
| `POST` | `/api/claims/process` | Full claim processing pipeline |
| `POST` | `/api/claims/classify` | Classify claim type & severity |
| `POST` | `/api/claims/extract` | Extract entities from description |
| `GET`  | `/api/data-quality/report` | Full data quality report |
| `POST` | `/api/data-quality/clean` | Clean & standardize data |
| `GET`  | `/api/analytics/summary` | Dashboard summary metrics |
| `GET`  | `/api/analytics/claims-by-type` | Claims distribution by type |
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

Top predictive features: `claim_amount`, `report_delay_days`, `credit_score`, `num_prior_claims`, `annual_premium`

---

## License

This project is for educational and demonstration purposes.
