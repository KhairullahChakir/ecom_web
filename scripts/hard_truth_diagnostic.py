import numpy as np
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Paths
DATA_DIR = "d:/op_ecom/data/processed_rigorous"

def diagnostic_bottom_line():
    print("Loading rigorous test data...")
    xp_te = np.load(os.path.join(DATA_DIR, "X_page_test.npy"))
    y_te = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    
    # 1. Trivial Baseline: Did they Add To Cart?
    # y=0 is Purchase, y=1 is Abandonment.
    # Our "Hueristic" for purchase is "Has Page Type 2"
    has_atc = np.any(xp_te == 2, axis=1).astype(int)
    
    # Flipping labels for standard AUC calculation (y=1 is Success)
    y_te_success = 1 - y_te
    heuristic_success = has_atc
    
    auc_atc = roc_auc_score(y_te_success, heuristic_success)
    prec, rec, _ = precision_recall_curve(y_te_success, heuristic_success)
    pr_auc_atc = auc(rec, prec)
    
    print("\n--- The Hard Truth Diagnostic ---")
    print(f"Heuristic Baseline (Predict Purchase if AddToCart exists):")
    print(f"ROC-AUC: {auc_atc:.4f}")
    print(f"PR-AUC:  {pr_auc_atc:.4f}")
    
    print("\nInsight: If ROC-AUC is > 0.90 just by spotting an 'AddToCart',")
    print("then Model 2 MUST beat this to be impressive.")

if __name__ == "__main__":
    diagnostic_bottom_line()
