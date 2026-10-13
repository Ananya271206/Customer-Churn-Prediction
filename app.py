import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction & Retention AI",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .high-risk {
        color: #DC2626;
        font-weight: bold;
    }
    .medium-risk {
        color: #D97706;
        font-weight: bold;
    }
    .low-risk {
        color: #16A34A;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

@st.cache_resource
def load_models_and_preprocessor():
    """Load cached preprocessor and trained models."""
    bundle_path = os.path.join(MODELS_DIR, "preprocessor_bundle.joblib")
    if not os.path.exists(bundle_path):
        return None, None
    bundle = joblib.load(bundle_path)
    
    models = {}
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        m_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if os.path.exists(m_path):
            models[name] = joblib.load(m_path)
            
    return bundle, models

@st.cache_data
def load_comparison_metrics():
    comp_path = os.path.join(REPORTS_DIR, "model_metrics_comparison.csv")
    if os.path.exists(comp_path):
        return pd.read_csv(comp_path)
    return None

@st.cache_data
def load_sample_dataset():
    data_path = os.path.join(BASE_DIR, "data", "raw", "telco_customer_churn.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

bundle, models = load_models_and_preprocessor()
metrics_df = load_comparison_metrics()
raw_df = load_sample_dataset()

# Title Section
st.markdown('<div class="main-header">📈 Telecom Customer Churn Prediction Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered early churn detection and retention recommendations using Logistic Regression, Random Forest, and XGBoost</div>', unsafe_allow_html=True)

# Top KPI Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers Evaluated", "7,043", "Telco Benchmark")
with col2:
    st.metric("Historical Churn Rate", "26.54%", "1,869 Churned")
with col3:
    st.metric("Best Model ROC-AUC", "0.845", "XGBoost")
with col4:
    st.metric("Peak Churn Recall", "80.21%", "Random Forest")

st.markdown("---")

# Main Navigation Tabs
tab_simulator, tab_eda, tab_benchmark, tab_batch = st.tabs([
    "🔮 Live Customer Risk Simulator",
    "📊 Exploratory Data Analysis (EDA)",
    "🤖 Model Benchmark & Evaluation",
    "📋 Batch Prediction & Dataset"
])

# ----------------- TAB 1: LIVE SIMULATOR -----------------
with tab_simulator:
    st.subheader("🎯 Real-Time Customer Churn Risk Prediction")
    st.markdown("Enter customer account details below to predict their likelihood of churning and view prescriptive retention actions.")
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    with sim_col1:
        st.markdown("##### 👤 Customer Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (Senior)" if x == 1 else "No (Non-Senior)")
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (Months with company)", min_value=0, max_value=72, value=4, help="Number of months customer has stayed with service")

    with sim_col2:
        st.markdown("##### 📡 Subscribed Services")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        mult_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        
        sec_options = ["No internet service"] if internet == "No" else ["No", "Yes"]
        security = st.selectbox("Online Security", sec_options)
        backup = st.selectbox("Online Backup", sec_options)
        protection = st.selectbox("Device Protection", sec_options)
        tech_support = st.selectbox("Tech Support", sec_options)

    with sim_col3:
        st.markdown("##### 💳 Contract & Billing")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=85.50, step=1.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly_charges), step=10.0)
        
        streaming_options = ["No internet service"] if internet == "No" else ["No", "Yes"]
        st_tv = st.selectbox("Streaming TV", streaming_options)
        st_movies = st.selectbox("Streaming Movies", streaming_options)

    # Model Selection for Simulation
    st.markdown("##### ⚙️ Prediction Model")
    selected_model_key = st.radio(
        "Choose Model for Inference:",
        ["xgboost", "random_forest", "logistic_regression"],
        format_func=lambda x: {"xgboost": "XGBoost (High Precision & Calibration)", 
                               "random_forest": "Random Forest (High Recall - 80.2%)", 
                               "logistic_regression": "Logistic Regression (Linear Benchmark)"}[x],
        horizontal=True
    )

    if st.button("🚀 Analyze Churn Risk", type="primary", use_container_width=True):
        if not bundle or not models:
            st.error("Models not loaded. Please ensure models are trained.")
        else:
            cust_data = {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": mult_lines,
                "InternetService": internet,
                "OnlineSecurity": security,
                "OnlineBackup": backup,
                "DeviceProtection": protection,
                "TechSupport": tech_support,
                "StreamingTV": st_tv,
                "StreamingMovies": st_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }
            
            # Preprocessing
            from src.preprocessing import clean_data
            df_in = pd.DataFrame([cust_data])
            df_cleaned = clean_data(df_in)
            
            preprocessor = bundle['preprocessor']
            feature_names = bundle['feature_names']
            
            X_trans = preprocessor.transform(df_cleaned)
            X_df = pd.DataFrame(X_trans, columns=feature_names)
            
            model = models[selected_model_key]
            prob = float(model.predict_proba(X_df)[0, 1])
            pred = int(model.predict(X_df)[0])
            
            # Display Prediction Results
            st.markdown("---")
            res_c1, res_c2, res_c3 = st.columns([1, 1, 2])
            
            with res_c1:
                st.markdown("#### Churn Probability")
                st.metric(label="Likelihood to Leave", value=f"{prob*100:.1f}%")
                st.progress(min(max(prob, 0.0), 1.0))
                
            with res_c2:
                st.markdown("#### Risk Status")
                if prob >= 0.65:
                    st.error("🚨 HIGH CHURN RISK")
                    st.markdown("**Status:** At Immediate Risk of Cancellation")
                elif prob >= 0.35:
                    st.warning("⚠️ MODERATE RISK")
                    st.markdown("**Status:** Showing Disengagement Signals")
                else:
                    st.success("✅ LOW RISK")
                    st.markdown("**Status:** Satisfied / Stable Customer")

            with res_c3:
                st.markdown("#### 💡 Prescriptive Retention Playbook")
                if prob >= 0.65:
                    st.markdown("""
                    - **Immediate Outreach:** Assign to VIP retention specialist within 24 hours.
                    - **Contract Migration:** Offer 15% discount for switching to an annual contract.
                    - **Support Package:** Add free Tech Support and Online Security trial for 6 months.
                    - **Payment Incentive:** Offer $10 statement credit for switching to automated bank/card payments.
                    """)
                elif prob >= 0.35:
                    st.markdown("""
                    - **Satisfaction Check:** Send automated NPS survey with follow-up call.
                    - **Bundle Exploration:** Offer bundled digital security or entertainment perk.
                    - **Usage Review:** Audit bandwidth quality and offer service optimization.
                    """)
                else:
                    st.markdown("""
                    - **Loyalty Reinforcement:** Send anniversary appreciation rewards.
                    - **Referral Campaign:** Customer is a strong candidate for advocacy promotions.
                    - **Maintain Quality:** Standard cadence updates.
                    """)

# ----------------- TAB 2: EDA INSIGHTS -----------------
with tab_eda:
    st.subheader("🔍 Exploratory Data Analysis & Churn Patterns")
    st.markdown("Data-driven discoveries across 7,043 customers revealing why and when churn happens.")
    
    eda_c1, eda_c2 = st.columns(2)
    with eda_c1:
        if os.path.exists(os.path.join(FIGURES_DIR, "churn_distribution.png")):
            st.image(os.path.join(FIGURES_DIR, "churn_distribution.png"), caption="Overall Customer Churn Ratio (26.5% vs 73.5%)", use_container_width=True)
        if os.path.exists(os.path.join(FIGURES_DIR, "services_churn_pattern.png")):
            st.image(os.path.join(FIGURES_DIR, "services_churn_pattern.png"), caption="Churn Rate by Internet Service and Tech Support", use_container_width=True)
            
    with eda_c2:
        if os.path.exists(os.path.join(FIGURES_DIR, "contract_churn_rate.png")):
            st.image(os.path.join(FIGURES_DIR, "contract_churn_rate.png"), caption="Contract Commitment vs Churn Probability", use_container_width=True)
        if os.path.exists(os.path.join(FIGURES_DIR, "tenure_charges_distribution.png")):
            st.image(os.path.join(FIGURES_DIR, "tenure_charges_distribution.png"), caption="KDE Density of Tenure and Monthly Charges by Churn", use_container_width=True)

    st.markdown("#### 📌 Key Churn Drivers Discovered:")
    st.info("""
    1. **Contract Duration is the #1 Predictor:** Customers on month-to-month contracts churn at **42.7%**, compared to only **2.8%** on two-year agreements.
    2. **Early Tenured Vulnerability:** The median tenure of churned customers is **10 months** versus **38 months** for retained clients. Customer churn is heaviest in the first 1-12 months.
    3. **Tech Support & Online Security as Anchors:** Customers without Tech Support churn at **41.6%**, while customers with Tech Support churn at just **15.2%**.
    4. **Fiber Optic Premium Expectation Gap:** Fiber optic customers have higher monthly charges ($70-110) and experience a **41.9%** churn rate, indicating price sensitivity or reliability dissatisfaction.
    """)

# ----------------- TAB 3: MODEL BENCHMARK -----------------
with tab_benchmark:
    st.subheader("🏆 Classification Model Benchmarking & Performance")
    st.markdown("Evaluation of Logistic Regression, Random Forest, and XGBoost on unseen test data (1,409 customers) with 5-fold Stratified Cross-Validation.")
    
    if metrics_df is not None:
        st.dataframe(
            metrics_df.style.highlight_max(subset=['Accuracy', 'Recall', 'ROC-AUC', '5-Fold CV ROC-AUC'], color='#D1FAE5')
                            .format({
                                'Accuracy': '{:.4f}',
                                'Recall': '{:.4f}',
                                'ROC-AUC': '{:.4f}',
                                'Precision': '{:.4f}',
                                'F1-Score': '{:.4f}',
                                '5-Fold CV ROC-AUC': '{:.4f}',
                                '5-Fold CV Recall': '{:.4f}'
                            }),
            use_container_width=True
        )
        
    bench_c1, bench_c2 = st.columns(2)
    with bench_c1:
        if os.path.exists(os.path.join(FIGURES_DIR, "roc_curves_comparison.png")):
            st.image(os.path.join(FIGURES_DIR, "roc_curves_comparison.png"), caption="Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)
    with bench_c2:
        if os.path.exists(os.path.join(FIGURES_DIR, "confusion_matrices.png")):
            st.image(os.path.join(FIGURES_DIR, "confusion_matrices.png"), caption="Confusion Matrices Across All Models", use_container_width=True)

    st.markdown("#### 🔬 Model Feature Importances")
    feat_c1, feat_c2 = st.columns(2)
    with feat_c1:
        if os.path.exists(os.path.join(FIGURES_DIR, "xgb_feature_importance.png")):
            st.image(os.path.join(FIGURES_DIR, "xgb_feature_importance.png"), caption="XGBoost Top Feature Importances", use_container_width=True)
    with feat_c2:
        if os.path.exists(os.path.join(FIGURES_DIR, "rf_feature_importance.png")):
            st.image(os.path.join(FIGURES_DIR, "rf_feature_importance.png"), caption="Random Forest Top Feature Importances", use_container_width=True)

    st.markdown("#### 💡 Evaluation Takeaways:")
    st.success("""
    - **Random Forest** achieved the highest test recall (**80.21%**), successfully flagging 300 out of 374 churners in the test holdout set.
    - **XGBoost** reached the best ROC-AUC score (**0.8446** test, **0.8469** 5-fold CV), providing exceptional ranking calibration for score thresholds.
    - **Logistic Regression** delivered competitive recall (**78.34%**) with full linear interpretability via odds ratios.
    """)

# ----------------- TAB 4: BATCH PREDICTION -----------------
with tab_batch:
    st.subheader("📋 Dataset Inspection & Sample Profiles")
    if raw_df is not None:
        st.markdown(f"**Loaded Raw Dataset:** {len(raw_df):,} customers across {raw_df.shape[1]} attributes.")
        st.dataframe(raw_df.head(15), use_container_width=True)
        
        st.download_button(
            label="📥 Download Cleaned Churn Data CSV",
            data=raw_df.to_csv(index=False),
            file_name="telco_customer_churn_cleaned.csv",
            mime="text/csv"
        )
