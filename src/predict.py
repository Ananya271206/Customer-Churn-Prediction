import os
import joblib
import pandas as pd
import numpy as np
from preprocessing import clean_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

class ChurnPredictor:
    """Production predictor for Customer Churn inference."""
    def __init__(self, model_name="xgboost"):
        bundle_path = os.path.join(MODELS_DIR, "preprocessor_bundle.joblib")
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Preprocessor bundle not found at {bundle_path}. Run train.py first.")
        
        self.bundle = joblib.load(bundle_path)
        self.preprocessor = self.bundle['preprocessor']
        self.feature_names = self.bundle['feature_names']
        
        model_file = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found at {model_file}")
        
        self.model = joblib.load(model_file)
        self.model_name = model_name

    def predict_customer(self, customer_dict: dict) -> dict:
        """
        Predict churn probability and risk level for a single customer record.
        """
        df_input = pd.DataFrame([customer_dict])
        cleaned_df = clean_data(df_input)
        
        # Transform using fitted preprocessor
        X_trans = self.preprocessor.transform(cleaned_df)
        X_df = pd.DataFrame(X_trans, columns=self.feature_names)
        
        prob = float(self.model.predict_proba(X_df)[0, 1])
        pred = int(self.model.predict(X_df)[0])
        
        if prob < 0.35:
            risk_level = "Low Risk"
            recommendation = "Customer is satisfied. Maintain standard loyalty rewards and periodic engagement."
        elif prob < 0.65:
            risk_level = "Moderate Risk"
            recommendation = "Early warning sign. Proactively offer tech support assistance or bundled service discounts."
        else:
            risk_level = "High Risk"
            recommendation = "URGENT INTERVENTION: Propose long-term contract migration discount, dedicated support specialist, or price-lock incentive."
            
        return {
            "model_used": self.model_name,
            "churn_prediction": "Likely to Churn" if pred == 1 else "Likely to Retain",
            "churn_probability": round(prob, 4),
            "churn_probability_pct": f"{prob * 100:.1f}%",
            "risk_level": risk_level,
            "retention_strategy": recommendation
        }

if __name__ == "__main__":
    # Test sample profile (high risk: month-to-month, fiber optic, no tech support, high monthly charges)
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 191.00
    }
    
    predictor = ChurnPredictor(model_name="xgboost")
    result = predictor.predict_customer(sample_customer)
    print("Inference Test Result for High-Risk Customer:")
    for k, v in result.items():
        print(f"  {k}: {v}")
