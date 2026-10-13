import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "telco_customer_churn.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")

def load_and_clean_data(filepath=DATA_PATH):
    """Load raw dataset and perform initial type conversion and cleaning."""
    df = pd.read_csv(filepath)
    # TotalCharges contains blank spaces for tenure == 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan), errors='coerce')
    # Impute missing TotalCharges with 0 (tenure is 0)
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    return df

def run_eda():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = load_and_clean_data()
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS: TELCO CUSTOMER CHURN")
    print("=" * 60)
    print(f"Total Customers: {len(df):,}")
    print(f"Features Count: {df.shape[1]}")
    churn_counts = df['Churn'].value_counts()
    churn_rate = (churn_counts['Yes'] / len(df)) * 100
    print(f"Retained Customers (No): {churn_counts['No']:,} ({100 - churn_rate:.2f}%)")
    print(f"Churned Customers (Yes): {churn_counts['Yes']:,} ({churn_rate:.2f}%)")
    print("-" * 60)

    # 1. Churn Class Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = ['#2b5c8f', '#d9534f']
    sns.countplot(data=df, x='Churn', hue='Churn', palette=palette, legend=False, ax=ax)
    ax.set_title("Customer Churn Distribution", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Churn Status", fontsize=12)
    ax.set_ylabel("Customer Count", fontsize=12)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:,}\n({height/len(df)*100:.1f}%)',
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    plt.tight_layout()
    dist_path = os.path.join(FIGURES_DIR, "churn_distribution.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"Saved: {dist_path}")

    # 2. Churn by Contract Type
    contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    contract_churn.columns = ['Contract', 'ChurnRate']
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = sns.barplot(data=contract_churn, x='Contract', y='ChurnRate', hue='Contract', palette='Reds_r', legend=False, ax=ax)
    ax.set_title("Churn Rate by Contract Type (%)", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Churn Rate (%)", fontsize=12)
    ax.set_xlabel("Contract Type", fontsize=12)
    ax.set_ylim(0, 50)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.1f}%',
                    (p.get_x() + p.get_width() / 2., height + 1),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    contract_path = os.path.join(FIGURES_DIR, "contract_churn_rate.png")
    plt.savefig(contract_path, dpi=300)
    plt.close()
    print(f"Saved: {contract_path}")

    # 3. Churn by Internet Service & Tech Support
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    is_churn = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    sns.barplot(data=is_churn, x='InternetService', y='Churn', hue='InternetService', palette='Blues_r', legend=False, ax=ax1)
    ax1.set_title("Churn Rate by Internet Service", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Churn Rate (%)", fontsize=11)
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.1f}%',
                     (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    ts_churn = df.groupby('TechSupport')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    sns.barplot(data=ts_churn, x='TechSupport', y='Churn', hue='TechSupport', palette='Purples_r', legend=False, ax=ax2)
    ax2.set_title("Churn Rate by Tech Support Status", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Churn Rate (%)", fontsize=11)
    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.1f}%',
                     (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    services_path = os.path.join(FIGURES_DIR, "services_churn_pattern.png")
    plt.savefig(services_path, dpi=300)
    plt.close()
    print(f"Saved: {services_path}")

    # 4. Tenure & Monthly Charges Distribution by Churn
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.kdeplot(data=df, x='tenure', hue='Churn', common_norm=False, fill=True, palette={'No': '#2b5c8f', 'Yes': '#d9534f'}, ax=ax1)
    ax1.set_title("Tenure Distribution by Churn Status", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Tenure (Months)", fontsize=11)
    
    sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', common_norm=False, fill=True, palette={'No': '#2b5c8f', 'Yes': '#d9534f'}, ax=ax2)
    ax2.set_title("Monthly Charges Distribution by Churn Status", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Monthly Charges ($)", fontsize=11)
    plt.tight_layout()
    tenure_path = os.path.join(FIGURES_DIR, "tenure_charges_distribution.png")
    plt.savefig(tenure_path, dpi=300)
    plt.close()
    print(f"Saved: {tenure_path}")

    # 5. Correlation Heatmap for Numerical Features + Target
    df_num = df[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
    df_num['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)
    corr = df_num.corr()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title("Correlation Heatmap: Numerical Features & Churn", fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    corr_path = os.path.join(FIGURES_DIR, "correlation_matrix.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved: {corr_path}")

    print("\nKEY CHURN PATTERNS IDENTIFIED:")
    print("1. Contract Type Impact:")
    print(f"   - Month-to-Month churn rate: {contract_churn.loc[contract_churn['Contract']=='Month-to-month', 'ChurnRate'].values[0]:.1f}%")
    print(f"   - Two-Year contract churn rate: {contract_churn.loc[contract_churn['Contract']=='Two year', 'ChurnRate'].values[0]:.1f}%")
    print("   -> Customers on flexible month-to-month contracts are ~15x more likely to churn than 2-year contracted clients.")
    
    print("2. Tenure Vulnerability:")
    churned_tenure_median = df[df['Churn'] == 'Yes']['tenure'].median()
    retained_tenure_median = df[df['Churn'] == 'No']['tenure'].median()
    print(f"   - Median tenure for churned customers: {churned_tenure_median:.0f} months")
    print(f"   - Median tenure for retained customers: {retained_tenure_median:.0f} months")
    print("   -> The highest churn risk window is during the first 1-12 months of onboarding.")

    print("3. Internet Service & Tech Support:")
    fo_churn = is_churn.loc[is_churn['InternetService'] == 'Fiber optic', 'Churn'].values[0]
    dsl_churn = is_churn.loc[is_churn['InternetService'] == 'DSL', 'Churn'].values[0]
    print(f"   - Fiber optic customer churn rate: {fo_churn:.1f}% vs DSL: {dsl_churn:.1f}%")
    no_ts_churn = ts_churn.loc[ts_churn['TechSupport'] == 'No', 'Churn'].values[0]
    has_ts_churn = ts_churn.loc[ts_churn['TechSupport'] == 'Yes', 'Churn'].values[0]
    print(f"   - No Tech Support churn rate: {no_ts_churn:.1f}% vs Has Tech Support: {has_ts_churn:.1f}%")
    print("   -> Premium fiber optic users without active tech support experience highest dissatisfaction.")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()
