from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_models(scale_pos_weight=2.76):
    """
    Returns the three classification models configured for churn prediction:
    1. Logistic Regression: Interpretable linear baseline with balanced class weights
    2. Random Forest: Bagging tree ensemble capturing non-linear interactions
    3. XGBoost: Gradient boosting algorithm with positive class re-weighting
    """
    models = {
        "Logistic Regression": LogisticRegression(
            C=0.5,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    }
    return models
