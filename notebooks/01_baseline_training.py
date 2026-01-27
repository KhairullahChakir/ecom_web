"""
Phase 1: Baseline Model Training
================================
Train Logistic Regression and XGBoost on UCI Online Shoppers dataset.
Generate metrics report (AUC, F1, Precision, Recall).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import xgboost as xgb
import joblib
import json
from datetime import datetime

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "online_shoppers_intention.csv")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "models")
REPORTS_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "metrics")

def load_and_preprocess_data():
    """Load and preprocess the UCI Online Shoppers dataset"""
    print("=" * 60)
    print("PHASE 1: BASELINE MODEL TRAINING")
    print("=" * 60)
    
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  → Dataset shape: {df.shape}")
    print(f"  → Columns: {list(df.columns)}")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f"  → Missing values: {missing}")
    
    # Target distribution
    print(f"\n  → Target distribution:")
    print(f"     Revenue=True:  {(df['Revenue'] == True).sum()} ({(df['Revenue'] == True).mean()*100:.1f}%)")
    print(f"     Revenue=False: {(df['Revenue'] == False).sum()} ({(df['Revenue'] == False).mean()*100:.1f}%)")
    
    print("\n[2/5] Encoding categorical features...")
    # Separate features and target
    X = df.drop('Revenue', axis=1)
    y = df['Revenue'].astype(int)
    
    # Identify categorical columns
    categorical_cols = ['Month', 'VisitorType', 'Weekend']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    # Label encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"  → Encoded {col}: {le.classes_}")
    
    print("\n[3/5] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  → Train set: {X_train.shape[0]} samples")
    print(f"  → Test set:  {X_test.shape[0]} samples")
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(X.columns),
        'scaler': scaler,
        'label_encoders': label_encoders,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols
    }


def train_logistic_regression(data):
    """Train Logistic Regression baseline"""
    print("\n[4/5] Training Logistic Regression...")
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced data
    )
    model.fit(data['X_train'], data['y_train'])
    
    # Predictions
    y_pred = model.predict(data['X_test'])
    y_prob = model.predict_proba(data['X_test'])[:, 1]
    
    # Metrics
    metrics = {
        'model': 'Logistic Regression',
        'auc_roc': roc_auc_score(data['y_test'], y_prob),
        'f1': f1_score(data['y_test'], y_pred),
        'precision': precision_score(data['y_test'], y_pred),
        'recall': recall_score(data['y_test'], y_pred)
    }
    
    print(f"  → AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  → F1 Score:  {metrics['f1']:.4f}")
    print(f"  → Precision: {metrics['precision']:.4f}")
    print(f"  → Recall:    {metrics['recall']:.4f}")
    
    return model, metrics


def train_xgboost(data):
    """Train XGBoost baseline"""
    print("\n[5/5] Training XGBoost...")
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (data['y_train'] == 0).sum() / (data['y_train'] == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(data['X_train'], data['y_train'])
    
    # Predictions
    y_pred = model.predict(data['X_test'])
    y_prob = model.predict_proba(data['X_test'])[:, 1]
    
    # Metrics
    metrics = {
        'model': 'XGBoost',
        'auc_roc': roc_auc_score(data['y_test'], y_prob),
        'f1': f1_score(data['y_test'], y_pred),
        'precision': precision_score(data['y_test'], y_pred),
        'recall': recall_score(data['y_test'], y_pred)
    }
    
    print(f"  → AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  → F1 Score:  {metrics['f1']:.4f}")
    print(f"  → Precision: {metrics['precision']:.4f}")
    print(f"  → Recall:    {metrics['recall']:.4f}")
    
    # Feature importance
    importance = dict(zip(data['feature_names'], model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  → Top 5 Features:")
    for feat, imp in top_features:
        print(f"     {feat}: {imp:.4f}")
    
    return model, metrics


def save_results(lr_model, lr_metrics, xgb_model, xgb_metrics, data):
    """Save models and metrics report"""
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(REPORTS_PATH, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Save models
    joblib.dump(lr_model, os.path.join(MODELS_PATH, 'logistic_regression.joblib'))
    joblib.dump(xgb_model, os.path.join(MODELS_PATH, 'xgboost.joblib'))
    joblib.dump(data['scaler'], os.path.join(MODELS_PATH, 'scaler.joblib'))
    joblib.dump(data['label_encoders'], os.path.join(MODELS_PATH, 'label_encoders.joblib'))
    print(f"\n  → Models saved to: {MODELS_PATH}")
    
    # Save metrics report
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'name': 'UCI Online Shoppers Purchasing Intention',
            'samples': 12330,
            'features': 17,
            'target': 'Revenue',
            'test_size': 0.2
        },
        'models': [lr_metrics, xgb_metrics]
    }
    
    report_path = os.path.join(REPORTS_PATH, 'baseline_metrics.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  → Metrics saved to: {report_path}")
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    print(f"\n{'Model':<25} {'AUC-ROC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 65)
    for m in [lr_metrics, xgb_metrics]:
        print(f"{m['model']:<25} {m['auc_roc']:>10.4f} {m['f1']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f}")
    
    # Winner
    winner = 'XGBoost' if xgb_metrics['auc_roc'] > lr_metrics['auc_roc'] else 'Logistic Regression'
    print(f"\n  🏆 Best baseline: {winner} (higher AUC-ROC)")
    
    return report


def main():
    """Main training pipeline"""
    # Load and preprocess
    data = load_and_preprocess_data()
    
    # Train baselines
    lr_model, lr_metrics = train_logistic_regression(data)
    xgb_model, xgb_metrics = train_xgboost(data)
    
    # Save results
    report = save_results(lr_model, lr_metrics, xgb_model, xgb_metrics, data)
    
    print("\n" + "=" * 60)
    print("✅ PHASE 1 COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  → Phase 2: Train TabM deep learning model")
    print("  → Phase 3: CPU latency benchmarking")
    
    return report


if __name__ == "__main__":
    main()
