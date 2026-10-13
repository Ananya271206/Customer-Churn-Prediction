import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

def calculate_metrics(y_true, y_pred, y_prob):
    """Computes full suite of classification metrics focusing on Accuracy, Recall, and ROC-AUC."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Recall": float(recall_score(y_true, y_pred)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "Precision": float(precision_score(y_true, y_pred)),
        "F1-Score": float(f1_score(y_true, y_pred))
    }

def plot_roc_curves(results, y_test, save_path=None):
    """Plots comparative ROC curves for all models."""
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for (name, data), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, data['y_prob'])
        auc_val = data['metrics']['ROC-AUC']
        plt.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} (ROC-AUC = {auc_val:.3f})")
        
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved ROC Curves: {save_path}")
    plt.close()

def plot_confusion_matrices(results, y_test, save_path=None):
    """Plots confusion matrices side-by-side for all evaluated models."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]
        
    for ax, (name, data) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, data['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Retained', 'Churned'],
                    yticklabels=['Retained', 'Churned'],
                    annot_kws={"size": 13, "weight": "bold"})
        ax.set_title(f"{name}\nRecall: {data['metrics']['Recall']:.3f} | AUC: {data['metrics']['ROC-AUC']:.3f}",
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved Confusion Matrices: {save_path}")
    plt.close()

def plot_feature_importances(model, feature_names, model_name, top_n=15, save_path=None):
    """Extracts and plots top feature importances or coefficients."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        title = f"Top {top_n} Feature Importances ({model_name})"
        x_label = "Relative Importance"
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        title = f"Top {top_n} Absolute Coefficients ({model_name})"
        x_label = "|Coefficient Magnitude|"
    else:
        return

    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=feat_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.xlabel(x_label, fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved Feature Importances: {save_path}")
    plt.close()
    return feat_df
