import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Paths
DATA_DIR = "d:/op_ecom/data/processed_rigorous"

def load_and_aggregate(name):
    xp = np.load(os.path.join(DATA_DIR, f"X_page_{name}.npy"))
    xd = np.load(os.path.join(DATA_DIR, f"X_dur_{name}.npy"))
    y = np.load(os.path.join(DATA_DIR, f"y_{name}.npy"))
    
    # Simple aggregates
    n_views = np.sum(xp == 1, axis=1)
    n_atc = np.sum(xp == 2, axis=1)
    total_dur = np.sum(xd, axis=1)
    seq_len = np.sum(xp > 0, axis=1)
    
    X = np.column_stack([n_views, n_atc, total_dur, seq_len])
    return X, y

def run_xgboost_baseline():
    print("Loading rigorous data for XGBoost baseline...")
    X_train, y_train = load_and_aggregate("train")
    X_test, y_test = load_and_aggregate("test")
    
    print("Training XGBoost...")
    # y=0 is Purchase, y=1 is Abandon. XGBoost likes y=1 as success.
    # Let's flip for XGBoost internal logic and then flip back for metrics.
    y_tr_flip = 1 - y_train 
    y_te_flip = 1 - y_test
    
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1, scale_pos_weight=100)
    model.fit(X_train, y_tr_flip)
    
    probs = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_te_flip, probs)
    prec, rec, _ = precision_recall_curve(y_te_flip, probs)
    pr_auc = auc(rec, prec)
    
    print("\n--- XGBoost Rigorous Baseline (Buyer Detection) ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    
if __name__ == "__main__":
    run_xgboost_baseline()
