# 📉 Customer Churn Prediction & Retention Engine

An end-to-end Machine Learning pipeline and interactive AI application designed to identify customers at risk of churning, pinpoint root causes, and prescribe proactive retention strategies before cancellations occur.

---

## 🎯 Executive Summary & Objectives

- **Primary Goal:** Accurately classify which subscribers are likely to terminate service.
- **Dataset:** Industry-standard **IBM Telco Customer Churn** dataset (7,043 subscriber accounts, 21 attributes).
- **Core Models:** 
  1. **Logistic Regression** (Interpretable linear baseline)
  2. **Random Forest** (Non-linear bagging ensemble)
  3. **XGBoost** (Gradient boosted decision trees)
- **Primary Evaluation Metrics:** **Accuracy**, **Recall** (capturing churning customers), and **ROC-AUC** (discrimination calibration).

---

## 📊 Key Findings from Exploratory Data Analysis (EDA)

Through exploratory data analysis across 7,043 customer records, the following churn patterns were uncovered:

| Behavioral Dimension | Churn Pattern Observed | Business Implication |
| :--- | :--- | :--- |
| **Contract Duration** | **Month-to-month:** 42.7% churn<br>**One-year:** 11.3% churn<br>**Two-year:** 2.8% churn | Flexible month-to-month contracts exhibit ~15x higher churn rate than 2-year contracted clients. |
| **Customer Tenure** | Churned median tenure: **10 months**<br>Retained median tenure: **38 months** | Highest attrition occurs within the first 1–12 months of onboarding. |
| **Tech Support & Security** | No Tech Support: **41.6%** churn<br>With Tech Support: **15.2%** churn | Customers with support add-ons have a strong psychological and functional anchor. |
| **Internet Service** | Fiber Optic: **41.9%** churn<br>DSL: **19.0%** churn | High pricing combined with service interruptions creates dissatisfaction among fiber optic users. |
| **Payment Method** | Electronic Check: **45.3%** churn<br>Credit Card / Bank Auto: **16.7%** churn | Friction in manual monthly payments increases cancellation propensity. |

---

## 🏆 Model Evaluation & Benchmark Results

All models were evaluated on an **unseen holdout test set (1,409 customers)** using **5-Fold Stratified Cross-Validation**:

| Model | Test Accuracy | Test Recall | Test ROC-AUC | Precision | F1-Score | 5-Fold CV ROC-AUC | 5-Fold CV Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **75.59%** | **80.21%** 🥇 | **0.8435** | **0.5263** | **0.6356** 🥇 | **0.8468** | **78.13%** |
| **XGBoost** | **74.80%** | **78.88%** | **0.8446** 🥇 | **0.5166** | **0.6243** | **0.8469** 🥇 | **78.86%** |
| **Logistic Regression** | **73.81%** | **78.34%** | **0.8418** | **0.5043** | **0.6136** | **0.8454** | **80.07%** |

### 🔍 Performance Takeaways:
- **Recall Priority:** In customer retention, False Negatives (missing a churning customer) are significantly more expensive than False Positives (sending a retention coupon to a happy customer). **Random Forest achieved 80.21% Recall**, catching 300 out of 374 churners.
- **ROC-AUC Champion:** **XGBoost achieved an ROC-AUC of 0.8446 (and 0.8469 in 5-fold CV)**, exhibiting the highest discrimination capability across all decision thresholds.
- **Top 5 Predictive Features (XGBoost):**
  1. `Contract_Two year`
  2. `InternetService_Fiber optic`
  3. `Contract_One year`
  4. `tenure`
  5. `MonthlyCharges`

---

## 📂 Project Architecture

```
Customer Churn Prediction/
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.csv          # 7,043 customer records
│   └── processed/
│       ├── cleaned_churn_data.csv
│       ├── X_train.csv, X_test.csv           # Stratified train/test splits
│       └── y_train.csv, y_test.csv
├── models/
│   ├── logistic_regression.joblib            # Trained scikit-learn models
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── preprocessor_bundle.joblib            # Fitted ColumnTransformer & feature names
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb    # Visual EDA notebook
│   └── 02_model_training_and_evaluation.ipynb# Modeling, CV, and evaluation
├── reports/
│   ├── figures/
│   │   ├── churn_distribution.png
│   │   ├── contract_churn_rate.png
│   │   ├── services_churn_pattern.png
│   │   ├── tenure_charges_distribution.png
│   │   ├── correlation_matrix.png
│   │   ├── roc_curves_comparison.png
│   │   ├── confusion_matrices.png
│   │   ├── rf_feature_importance.png
│   │   └── xgb_feature_importance.png
│   └── model_metrics_comparison.csv
├── src/
│   ├── download_data.py                      # Data retrieval script
│   ├── eda.py                                # Statistical EDA and plot generation
│   ├── preprocessing.py                      # Preprocessing & feature engineering
│   ├── models.py                             # Model configurations
│   ├── evaluate.py                           # Evaluation metrics & visualization
│   ├── train.py                              # Automated training & benchmark runner
│   ├── predict.py                            # Production inference engine
│   └── generate_notebooks.py                 # Notebook generation script
├── app.py                                    # Interactive Streamlit Web Application
├── requirements.txt                          # Python dependencies
└── README.md                                 # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Installation
Ensure Python 3.10+ is installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Full Training & Evaluation Pipeline
To re-run data preprocessing, train all 3 models with 5-fold CV, and generate all visual reports:
```bash
python src/train.py
```

### 3. Launch the Interactive Web Application & Simulator
To launch the interactive dashboard and real-time customer risk simulator:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Single Customer Prediction (CLI)
You can run standalone inference for any customer profile:
```bash
python src/predict.py
```

---

## 💡 Business Retention Playbook

Based on the model insights, the following retention interventions are recommended:

1. **New Customer Concierge Program (First 90 Days):**
   - Provide automated onboarding milestones and proactive satisfaction check-ins during the critical tenure window (< 12 months).
2. **Annual Contract Transition Offers:**
   - Offer a 10%–15% incentive for month-to-month subscribers to migrate to annual contracts.
3. **Bundled Tech Support Protection:**
   - Provide free or discounted Tech Support and Online Security to premium Fiber Optic subscribers to drastically reduce service friction.
4. **Automated Payment Incentive:**
   - Provide a one-time bill credit for customers transitioning from electronic/paper check to automated bank or card billing.
