# 🏦 Enterprise Fraud Risk Intelligence Platform

### Explainable & Drift-Aware Transaction Fraud Detection System

------------------------------------------------------------------------

## 📌 Overview

The Enterprise Fraud Risk Intelligence Platform is an advanced machine
learning system designed to detect fraudulent financial transactions in
real-time while ensuring transparency, reliability, and regulatory
compliance.

Unlike traditional fraud detection systems that only output a binary
decision (Fraud / Not Fraud), this platform:

-   Generates a **Fraud Risk Score (0–100%)**
-   Provides **Per-Transaction Explainable AI** insights via SHAP
-   Detects **Data & Model Drift** (PSI + KS tests)
-   Monitors **Model Health** continuously
-   Offers an **Interactive Streamlit Dashboard**

This project simulates a production-grade BFSI (Banking, Financial
Services & Insurance) fraud intelligence system.

------------------------------------------------------------------------

## 🎯 Problem Statement

Financial institutions process millions of transactions daily including:

-   Credit/Debit card payments
-   Online transfers
-   Digital wallet transactions
-   International payments

Fraudulent transactions are mixed within legitimate ones.

### ❗ Current Industry Challenges

1.  Fraud detected after money loss
2.  Black-box ML systems without explanations
3.  Models becoming outdated due to behaviour changes
4.  Lack of continuous monitoring

------------------------------------------------------------------------

## 💡 Solution

This platform provides:

-   **Early fraud prediction** with probability scoring (0–100%)
-   **Explainable AI** — per-transaction SHAP waterfall + narrative explanations
-   **Threshold optimisation** for business-configurable risk levels
-   **Drift monitoring** — PSI & Kolmogorov–Smirnov tests on every feature
-   **Model health dashboard** — tracks performance decay vs baseline
-   **Interactive Streamlit dashboard** for analysts and stakeholders

------------------------------------------------------------------------

## 🏗 System Architecture

```
Transaction Data
  → Data Preprocessing (SMOTE, RobustScaler, feature engineering)
  → XGBoost Fraud Model (tuned with RandomizedSearchCV + early stopping)
  → Fraud Risk Score (0–100%)
  → SHAP Explanation Engine (global + per-transaction)
  → Drift Monitor (PSI, KS-test, prediction drift)
  → Risk Intelligence Dashboard (Streamlit)
```

------------------------------------------------------------------------

## ⚙️ Key Features

### 1️⃣ Fraud Risk Scoring Engine (`fraud_scorer.py`)

Instead of binary classification, the model outputs:

**Fraud Risk Score (0–100%)**

| Risk Level | Threshold | Action |
|------------|-----------|--------|
| 🔴 High   | ≥ 70%     | **BLOCK** — immediate investigation |
| 🟡 Medium | 30–70%    | **REVIEW** — flag for analyst |
| 🟢 Low    | < 30%     | **ALLOW** — legitimate |

Features:
-   Per-transaction scoring with risk categorisation
-   SHAP-based feature-level explanations with human-readable narratives
-   Batch scoring with DataFrame output
-   Configurable risk thresholds

------------------------------------------------------------------------

### 2️⃣ Advanced XGBoost Model (`XGBoost.py`)

-   **Hyperparameter tuning** — RandomizedSearchCV (30 iterations, PR-AUC optimised)
-   **Early stopping** — monitors validation set to prevent overfitting
-   **Threshold optimisation** — fine-grained sweep (0.01–0.99, step 0.005)
-   **Class imbalance** — handled by SMOTE in preprocessing (no double-weighting)
-   **Baseline comparison** — Logistic Regression benchmark included

------------------------------------------------------------------------

### 3️⃣ Comprehensive Evaluation Metrics

-   ROC-AUC & PR-AUC
-   Precision, Recall, F1 Score
-   Confusion Matrix
-   ROC Curve & Precision-Recall Curve
-   Score Distribution (Legit vs Fraud)

------------------------------------------------------------------------

### 4️⃣ Explainable AI — SHAP (`fraud_scorer.py` + Dashboard)

**Global** — Summary plot showing which features matter most across all transactions.

**Per-Transaction** — For each scored transaction:
-   Top contributing features with SHAP values
-   Direction of influence (increases/decreases fraud risk)
-   Waterfall plot visualisation
-   Human-readable narrative explanation

Example output:
```
Fraud risk is HIGH (87.3%) because:
  ↑ V14 = -5.2341 (increases fraud risk)
  ↑ V4  = 3.1287 (increases fraud risk)
  ↓ V12 = -1.0543 (decreases fraud risk)
```

Ensures **regulatory compliance** and **transparency**.

------------------------------------------------------------------------

### 5️⃣ Drift Monitoring (`drift_monitor.py`)

Detects distribution shifts that signal model degradation:

-   **Population Stability Index (PSI)** — per feature
-   **Kolmogorov–Smirnov test** — statistical significance
-   **Prediction distribution drift** — val vs test
-   **Performance decay tracking** — current metrics vs baseline

Severity levels:
| PSI Value | Status | Action |
|-----------|--------|--------|
| < 0.10    | ✅ Stable | No action needed |
| 0.10–0.25 | ⚠️ Moderate Drift | Monitor closely |
| > 0.25    | 🚨 Significant Drift | Retrain model |

------------------------------------------------------------------------

### 6️⃣ Risk Intelligence Dashboard (`app.py`)

Interactive **Streamlit** dashboard with 5 pages:

1.  **📊 Performance Overview** — KPIs, ROC/PR curves, confusion matrix, score distribution
2.  **🔍 Transaction Scorer** — real-time scoring (manual input or sample from test set)
3.  **💡 SHAP Explanations** — global summary + per-transaction waterfall plots
4.  **📈 Feature Importance** — interactive top-N feature ranking
5.  **🔄 Drift Monitor** — feature PSI chart, prediction drift, model health status

------------------------------------------------------------------------

## 📊 Model Performance

### Baseline (Logistic Regression)

| Metric  | Value |
|---------|-------|
| ROC-AUC | 0.967 |

### Tuned XGBoost

| Metric    | Value  |
|-----------|--------|
| ROC-AUC   | 0.973  |
| PR-AUC    | 0.819  |
| Precision | 91.1%  |
| Recall    | 75.8%  |
| F1 Score  | 0.828  |

------------------------------------------------------------------------

## 📁 Project Structure

```
fraud-risk-intelligence/
│
├── creditcard.csv                 # Raw dataset
├── preprocessed.py                # Data preprocessing pipeline (SMOTE, scaling, splits)
├── XGBoost.py                     # Model training, tuning & evaluation
├── fraud_scorer.py                # Risk scoring engine + SHAP explanations
├── drift_monitor.py               # Data & model drift detection
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── README.md
│
├── processed_data/
│   ├── X_train.npy / X_val.npy / X_test.npy
│   ├── y_train.npy / y_val.npy / y_test.npy
│   ├── feature_names.csv
│   ├── scaler.joblib
│   └── preprocessing_report.json
│
└── results/
    ├── xgboost_fraud_model.json   # Saved XGBoost model
    ├── results.json               # Metrics & best hyperparameters
    ├── roc_pr_curves.png
    ├── feature_importance.png
    ├── shap_summary.png
    ├── drift_psi_chart.png
    ├── prediction_drift.png
    └── drift_report.json
```

------------------------------------------------------------------------

## 🛠 Technologies Used

| Category         | Technologies |
|------------------|-------------|
| Core ML          | XGBoost, Scikit-learn |
| Explainability   | SHAP |
| Drift Detection  | SciPy (KS-test), PSI |
| Data Processing  | Pandas, NumPy, imbalanced-learn (SMOTE) |
| Visualization    | Matplotlib, Seaborn |
| Dashboard        | Streamlit |
| Language         | Python 3.11 |

------------------------------------------------------------------------

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data

```bash
python preprocessed.py
```

### 3. Train the Model

```bash
python XGBoost.py
```

### 4. Run Drift Analysis

```bash
python drift_monitor.py
```

### 5. Test the Risk Scorer

```bash
python fraud_scorer.py
```

### 6. Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**

------------------------------------------------------------------------

## 🏆 Why This Project Stands Out

-   **End-to-end production pipeline** — preprocessing → training → scoring → monitoring → dashboard
-   **Explainable AI** — per-transaction SHAP narratives (regulatory compliance ready)
-   **Drift-aware** — automated detection of data & model degradation
-   **Risk scoring engine** — configurable business thresholds (Block / Review / Allow)
-   **Interactive dashboard** — 5-page Streamlit app for analysts & stakeholders
-   **Imbalance-aware** — SMOTE resampling + PR-AUC optimisation
-   **Early stopping** — prevents overfitting on validation set

------------------------------------------------------------------------

## 🔮 Future Enhancements

-   REST API deployment (FastAPI) for real-time scoring
-   Docker containerisation for deployment
-   Automated retraining pipeline triggered by drift alerts
-   Database integration for transaction logging
-   Role-based access control on the dashboard

------------------------------------------------------------------------

## 📜 License

This project is intended for educational and research purposes.

------------------------------------------------------------------------

Developed as part of an Enterprise BFSI Fraud Risk Intelligence use case.
