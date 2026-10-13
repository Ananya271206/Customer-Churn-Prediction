import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

from preprocessing import prepare_and_split_data
from models import get_models
from evaluate import calculate_metrics, plot_roc_curves, plot_confusion_matrices, plot_feature_importances

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

def train_and_evaluate_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION: TRAINING PIPELINE")
    print("=" * 70)
    
    # 1. Prepare data
    X_train, X_test, y_train, y_test, preprocessor_bundle = prepare_and_split_data()
    feature_names = preprocessor_bundle['feature_names']
    
    # Calculate scale_pos_weight
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    print(f"Calculated minority class imbalance weight: {pos_weight:.2f}")
    
    # 2. Instantiate models
    models = get_models(scale_pos_weight=pos_weight)
    
    results = {}
    cv_summary = []
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print("-" * 70)
        print(f"Training Model: {name}...")
        
        # 5-Fold Stratified Cross-Validation for ROC-AUC and Recall
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
        cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall').mean()
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
        
        # Fit on full training set
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'metrics': metrics,
            'cv_metrics': {
                'CV ROC-AUC': cv_auc,
                'CV Recall': cv_recall,
                'CV Accuracy': cv_acc
            }
        }
        
        # Save model artifact
        model_filename = f"{name.lower().replace(' ', '_')}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(model, model_path)
        print(f"  -> Model saved: {model_path}")
        
        print(f"  Test Accuracy: {metrics['Accuracy']:.4f} | Recall: {metrics['Recall']:.4f} | ROC-AUC: {metrics['ROC-AUC']:.4f}")
        print(f"  5-Fold CV    : Accuracy: {cv_acc:.4f} | Recall: {cv_recall:.4f} | ROC-AUC: {cv_auc:.4f}")

    # 3. Create Comparison Table
    comparison_rows = []
    for name, data in results.items():
        m = data['metrics']
        cv = data['cv_metrics']
        comparison_rows.append({
            'Model': name,
            'Accuracy': round(m['Accuracy'], 4),
            'Recall': round(m['Recall'], 4),
            'ROC-AUC': round(m['ROC-AUC'], 4),
            'Precision': round(m['Precision'], 4),
            'F1-Score': round(m['F1-Score'], 4),
            '5-Fold CV ROC-AUC': round(cv['CV ROC-AUC'], 4),
            '5-Fold CV Recall': round(cv['CV Recall'], 4)
        })
    
    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = os.path.join(REPORTS_DIR, "model_metrics_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print("\n" + "=" * 70)
    print("FINAL MODEL BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    print(comp_df.to_string(index=False))
    print(f"\nSaved comparison metrics: {comp_csv_path}")

    # 4. Generate & Save Visualizations
    roc_path = os.path.join(FIGURES_DIR, "roc_curves_comparison.png")
    plot_roc_curves(results, y_test, save_path=roc_path)

    cm_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plot_confusion_matrices(results, y_test, save_path=cm_path)

    # Plot feature importances for Random Forest and XGBoost
    rf_feat_path = os.path.join(FIGURES_DIR, "rf_feature_importance.png")
    plot_feature_importances(results['Random Forest']['model'], feature_names, 'Random Forest', top_n=12, save_path=rf_feat_path)

    xgb_feat_path = os.path.join(FIGURES_DIR, "xgb_feature_importance.png")
    plot_feature_importances(results['XGBoost']['model'], feature_names, 'XGBoost', top_n=12, save_path=xgb_feat_path)
    
    lr_feat_path = os.path.join(FIGURES_DIR, "lr_coefficients.png")
    plot_feature_importances(results['Logistic Regression']['model'], feature_names, 'Logistic Regression', top_n=12, save_path=lr_feat_path)

    print("\nAll training and evaluation steps completed successfully!")
    return results, comp_df

if __name__ == "__main__":
    train_and_evaluate_all()
