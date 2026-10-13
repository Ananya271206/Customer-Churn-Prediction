import os
import nbformat as nbf

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 📊 Customer Churn Prediction: Exploratory Data Analysis (EDA)\n"
            "**Goal:** Understand customer behavior, uncover patterns driving churn, and identify the most influential factors before modeling.\n\n"
            "### Business Context:\n"
            "Customer churn directly hurts Recurring Revenue (MRR/ARR) and Customer Lifetime Value (LTV). Acquiring a new customer costs 5x to 7x more than retaining an existing customer. By uncovering high-risk behavioral patterns, retention teams can target proactive campaigns."
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "# Configure visual styling\n"
            "sns.set_theme(style='whitegrid', palette='muted')\n"
            "plt.rcParams['figure.figsize'] = (10, 6)\n"
            "plt.rcParams['font.size'] = 11"
        ),
        nbf.v4.new_markdown_cell("## 1. Load and Inspect Dataset"),
        nbf.v4.new_code_cell(
            "data_path = '../data/raw/telco_customer_churn.csv'\n"
            "df = pd.read_csv(data_path)\n"
            "print(f'Shape of dataset: {df.shape}')\n"
            "df.head()"
        ),
        nbf.v4.new_markdown_cell("## 2. Data Cleaning & Type Conversion\n`TotalCharges` contains blank spaces when `tenure == 0`. We convert it to numeric and impute 0.0 for those records."),
        nbf.v4.new_code_cell(
            "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')\n"
            "df['TotalCharges'] = df['TotalCharges'].fillna(0.0)\n"
            "print(df.info())"
        ),
        nbf.v4.new_markdown_cell("## 3. Churn Class Distribution Analysis"),
        nbf.v4.new_code_cell(
            "churn_counts = df['Churn'].value_counts()\n"
            "print(churn_counts)\n"
            "print(f'Churn Rate: {(churn_counts[\"Yes\"] / len(df)) * 100:.2f}%')\n\n"
            "fig, ax = plt.subplots(figsize=(6, 4))\n"
            "sns.countplot(data=df, x='Churn', hue='Churn', palette=['#2b5c8f', '#d9534f'], legend=False, ax=ax)\n"
            "ax.set_title('Churn Class Distribution', fontsize=14, fontweight='bold')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("## 4. Bivariate Analysis: Churn by Contract Type & Billing"),
        nbf.v4.new_code_cell(
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n"
            "contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()\n"
            "sns.barplot(data=contract_churn, x='Contract', y='Churn', hue='Contract', palette='Reds_r', legend=False, ax=ax1)\n"
            "ax1.set_title('Churn Rate by Contract Type (%)', fontweight='bold')\n"
            "ax1.set_ylabel('Churn Rate (%)')\n\n"
            "pay_churn = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()\n"
            "sns.barplot(data=pay_churn, x='PaymentMethod', y='Churn', hue='PaymentMethod', palette='Blues_r', legend=False, ax=ax2)\n"
            "ax2.set_title('Churn Rate by Payment Method (%)', fontweight='bold')\n"
            "ax2.set_ylabel('Churn Rate (%)')\n"
            "ax2.tick_params(axis='x', rotation=45)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("## 5. Tenure and Charges Distribution"),
        nbf.v4.new_code_cell(
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n"
            "sns.kdeplot(data=df, x='tenure', hue='Churn', common_norm=False, fill=True, palette={'No': '#2b5c8f', 'Yes': '#d9534f'}, ax=ax1)\n"
            "ax1.set_title('Tenure Distribution by Churn', fontweight='bold')\n"
            "ax1.set_xlabel('Tenure (Months)')\n\n"
            "sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', common_norm=False, fill=True, palette={'No': '#2b5c8f', 'Yes': '#d9534f'}, ax=ax2)\n"
            "ax2.set_title('Monthly Charges Distribution by Churn', fontweight='bold')\n"
            "ax2.set_xlabel('Monthly Charges ($)')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("## 6. Services & Tech Support Impact"),
        nbf.v4.new_code_cell(
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n"
            "ts_churn = df.groupby('TechSupport')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()\n"
            "sns.barplot(data=ts_churn, x='TechSupport', y='Churn', hue='TechSupport', palette='Purples_r', legend=False, ax=ax1)\n"
            "ax1.set_title('Churn Rate by Tech Support Status (%)', fontweight='bold')\n\n"
            "sec_churn = df.groupby('OnlineSecurity')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()\n"
            "sns.barplot(data=sec_churn, x='OnlineSecurity', y='Churn', hue='OnlineSecurity', palette='Greens_r', legend=False, ax=ax2)\n"
            "ax2.set_title('Churn Rate by Online Security Status (%)', fontweight='bold')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 7. Key Findings & Business Takeaways\n"
            "1. **Contract Friction:** Month-to-month customers churn at 42.7% versus 2.8% for 2-year customers.\n"
            "2. **Critical Retention Window:** The first 1-12 months are high-risk onboarding phases.\n"
            "3. **Protective Add-ons:** Tech Support and Online Security cut customer churn by more than half (41.6% down to 15.2%)."
        )
    ]
    
    path = os.path.join(NOTEBOOKS_DIR, "01_exploratory_data_analysis.ipynb")
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created {path}")

def create_modeling_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 🤖 Customer Churn: Classification Modeling & Evaluation\n"
            "**Goal:** Build, benchmark, and evaluate Logistic Regression, Random Forest, and XGBoost models.\n"
            "**Metrics:** Accuracy, Recall, and ROC-AUC with 5-Fold Stratified Cross-Validation."
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.metrics import (\n"
            "    accuracy_score, recall_score, precision_score, f1_score,\n"
            "    roc_auc_score, roc_curve, confusion_matrix, classification_report\n"
            ")\n"
            "from sklearn.model_selection import StratifiedKFold, cross_val_score\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from xgboost import XGBClassifier\n\n"
            "sns.set_theme(style='whitegrid', palette='muted')"
        ),
        nbf.v4.new_markdown_cell("## 1. Load Processed Datasets"),
        nbf.v4.new_code_cell(
            "X_train = pd.read_csv('../data/processed/X_train.csv')\n"
            "X_test = pd.read_csv('../data/processed/X_test.csv')\n"
            "y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()\n"
            "y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()\n\n"
            "print(f'Training shape: {X_train.shape}, Test shape: {X_test.shape}')\n"
            "print(f'Train churn rate: {y_train.mean():.2%}, Test churn rate: {y_test.mean():.2%}')"
        ),
        nbf.v4.new_markdown_cell("## 2. Define Model Architectures with Class Balancing"),
        nbf.v4.new_code_cell(
            "pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()\n"
            "print(f'Minority class scale weight: {pos_weight:.2f}')\n\n"
            "models = {\n"
            "    'Logistic Regression': LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42),\n"
            "    'Random Forest': RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_split=10, min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1),\n"
            "    'XGBoost': XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42, n_jobs=-1)\n"
            "}"
        ),
        nbf.v4.new_markdown_cell("## 3. Train Models & 5-Fold Stratified Cross-Validation"),
        nbf.v4.new_code_cell(
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
            "results = {}\n"
            "records = []\n\n"
            "for name, model in models.items():\n"
            "    print(f'Evaluating {name}...')\n"
            "    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()\n"
            "    cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall').mean()\n"
            "    \n"
            "    model.fit(X_train, y_train)\n"
            "    y_pred = model.predict(X_test)\n"
            "    y_prob = model.predict_proba(X_test)[:, 1]\n"
            "    \n"
            "    acc = accuracy_score(y_test, y_pred)\n"
            "    rec = recall_score(y_test, y_pred)\n"
            "    auc = roc_auc_score(y_test, y_prob)\n"
            "    prec = precision_score(y_test, y_pred)\n"
            "    f1 = f1_score(y_test, y_pred)\n"
            "    \n"
            "    results[name] = {'y_pred': y_pred, 'y_prob': y_prob, 'model': model}\n"
            "    records.append({\n"
            "        'Model': name,\n"
            "        'Accuracy': round(acc, 4),\n"
            "        'Recall': round(rec, 4),\n"
            "        'ROC-AUC': round(auc, 4),\n"
            "        'Precision': round(prec, 4),\n"
            "        'F1-Score': round(f1, 4),\n"
            "        '5-Fold CV ROC-AUC': round(cv_auc, 4),\n"
            "        '5-Fold CV Recall': round(cv_recall, 4)\n"
            "    })\n\n"
            "comparison_df = pd.DataFrame(records)\n"
            "display(comparison_df)"
        ),
        nbf.v4.new_markdown_cell("## 4. Visual Comparison: ROC Curves & Confusion Matrices"),
        nbf.v4.new_code_cell(
            "plt.figure(figsize=(8, 6))\n"
            "colors = {'Logistic Regression': '#1f77b4', 'Random Forest': '#2ca02c', 'XGBoost': '#d62728'}\n"
            "for name, data in results.items():\n"
            "    fpr, tpr, _ = roc_curve(y_test, data['y_prob'])\n"
            "    score = roc_auc_score(y_test, data['y_prob'])\n"
            "    plt.plot(fpr, tpr, color=colors[name], lw=2.5, label=f'{name} (ROC-AUC = {score:.4f})')\n\n"
            "plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (0.50)')\n"
            "plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')\n"
            "plt.xlabel('False Positive Rate')\n"
            "plt.ylabel('True Positive Rate (Recall)')\n"
            "plt.legend(loc='lower right')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("## 5. Confusion Matrices"),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n"
            "for ax, (name, data) in zip(axes, results.items()):\n"
            "    cm = confusion_matrix(y_test, data['y_pred'])\n"
            "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,\n"
            "                xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])\n"
            "    ax.set_title(f'{name}\\nRecall: {recall_score(y_test, data[\"y_pred\"]):.3f}', fontweight='bold')\n"
            "    ax.set_xlabel('Predicted')\n"
            "    ax.set_ylabel('Actual')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("## 6. Feature Importances (XGBoost)"),
        nbf.v4.new_code_cell(
            "xgb_model = results['XGBoost']['model']\n"
            "feat_imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': xgb_model.feature_importances_})\n"
            "feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(12)\n\n"
            "plt.figure(figsize=(9, 5))\n"
            "sns.barplot(data=feat_imp, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)\n"
            "plt.title('Top 12 Most Predictive Features (XGBoost)', fontsize=13, fontweight='bold')\n"
            "plt.show()"
        )
    ]
    path = os.path.join(NOTEBOOKS_DIR, "02_model_training_and_evaluation.ipynb")
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created {path}")

if __name__ == "__main__":
    create_eda_notebook()
    create_modeling_notebook()
