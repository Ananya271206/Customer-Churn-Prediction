import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "telco_customer_churn.csv")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw dataframe: fixes TotalCharges, drops customerID, encodes target."""
    df_clean = df.copy()
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
    
    # Handle TotalCharges missing/whitespace
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'].replace(" ", np.nan), errors='coerce')
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0.0)

    # Feature Engineering
    # 1. Number of online services subscribed
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_clean['NumOnlineServices'] = (df_clean[service_cols] == 'Yes').sum(axis=1)
    
    # 2. Average monthly charges estimate
    df_clean['AvgMonthlyChargeEstimate'] = np.where(df_clean['tenure'] > 0, df_clean['TotalCharges'] / df_clean['tenure'], df_clean['MonthlyCharges'])
    
    # Target encoding if present
    if 'Churn' in df_clean.columns:
        df_clean['Churn'] = (df_clean['Churn'] == 'Yes').astype(int)
        
    return df_clean

def get_preprocessor(categorical_features, numeric_features):
    """Creates a scikit-learn ColumnTransformer for numeric and categorical pipelines."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

def prepare_and_split_data(raw_csv_path=RAW_DATA_PATH, test_size=0.2, random_state=42):
    """Loads, cleans, engineers features, splits data, fits preprocessor, and saves artifacts."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    df = pd.read_csv(raw_csv_path)
    df_cleaned = clean_data(df)
    
    # Save cleaned full dataset
    cleaned_csv_path = os.path.join(PROCESSED_DATA_DIR, "cleaned_churn_data.csv")
    df_cleaned.to_csv(cleaned_csv_path, index=False)
    
    X = df_cleaned.drop(columns=['Churn'])
    y = df_cleaned['Churn']
    
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'NumOnlineServices', 'AvgMonthlyChargeEstimate']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    preprocessor = get_preprocessor(categorical_features, numeric_features)
    preprocessor.fit(X_train_raw)
    
    # Extract transformed feature names
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = list(numeric_features) + list(cat_feature_names)
    
    X_train_transformed = preprocessor.transform(X_train_raw)
    X_test_transformed = preprocessor.transform(X_test_raw)
    
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test_raw.index)
    
    # Save processed datasets
    X_train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_train.csv"), index=False)
    X_test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, "y_test.csv"), index=False)
    
    # Save preprocessor and raw column definitions
    preprocessor_bundle = {
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }
    joblib.dump(preprocessor_bundle, os.path.join(MODELS_DIR, "preprocessor_bundle.joblib"))
    
    print(f"Data prepared successfully:")
    print(f"  Training set: {X_train_df.shape[0]} samples, {X_train_df.shape[1]} features")
    print(f"  Testing set:  {X_test_df.shape[0]} samples, {X_test_df.shape[1]} features")
    print(f"  Churn distribution - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
    
    return X_train_df, X_test_df, y_train, y_test, preprocessor_bundle

if __name__ == "__main__":
    prepare_and_split_data()
