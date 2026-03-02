import torch
import torch.nn as nn
import numpy as np
import os
import sys
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import train_test_split

# Add scripts dir for architecture
sys.path.append(os.path.abspath("scripts"))
from train_tcn import AbandonmentTCN

# Paths
DATA_DIR = "data/processed"
MODEL_PATH = "backend/models/exit_model_tcn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_minority_class():
    print(f"🔍 Starting Deep Evaluation on {DEVICE}...")
    
    # 1. Load Full Data
    X_page = np.load(os.path.join(DATA_DIR, "X_page_real.npy"))
    X_dur = np.load(os.path.join(DATA_DIR, "X_dur_real.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_abandon_real.npy"))
    
    # 2. Re-create the 80/10/10 Test Set (using same seed as training)
    _, temp_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    _, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=y[temp_idx], random_state=42)
    
    X_p_te, X_d_te, y_te = X_page[test_idx], X_dur[test_idx], y[test_idx]
    print(f"📊 Test Samples: {len(y_te):,}")
    
    # 3. Load Model
    model = AbandonmentTCN(num_page_types=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 4. Inference
    all_probs = []
    batch_size = 4096
    with torch.no_grad():
        for i in range(0, len(X_p_te), batch_size):
            p = torch.from_numpy(X_p_te[i:i+batch_size]).long().to(DEVICE)
            d = torch.from_numpy(X_d_te[i:i+batch_size]).float().to(DEVICE)
            logits = model(p, d).squeeze()
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
    
    y_probs_aband = np.array(all_probs)
    
    # CRITICAL: Flip to Purchase Class (Minority)
    # y=1 is Abandon, y=0 is Purchase.
    # We want to predict y=0.
    y_te_purch = (y_te == 0).astype(int)
    y_probs_purch = 1 - y_probs_aband # Probability of Purchase
    
    # 5. Metrics for Purchase (The "Hard Exam")
    prec, rec, _ = precision_recall_curve(y_te_purch, y_probs_purch)
    pr_auc_purch = auc(rec, prec)
    roc_auc_purch = roc_auc_score(y_te_purch, y_probs_purch)
    
    # Precision @ Top 1% (Finding the needle)
    n = len(y_probs_purch)
    k = max(int(n * 0.01), 1)
    top_k_idx = np.argsort(y_probs_purch)[::-1][:k]
    prec_at_1 = np.mean(y_te_purch[top_k_idx])
    
    # Confusion Matrix at 0.5 threshold for Purchase
    y_pred_purch = (y_probs_purch > 0.5).astype(int)
    cm = confusion_matrix(y_te_purch, y_pred_purch)
    
    print("\n" + "="*45)
    print("      DEEP EVALUATION: FINDING BUYERS")
    print("="*45)
    print(f"Target (Purchase) Rate: {y_te_purch.mean()*100:.2f}%")
    print(f"PR-AUC (Buyers):       {pr_auc_purch:.4f}")
    print(f"ROC-AUC (Buyers):      {roc_auc_purch:.4f}")
    print("-"*45)
    print(f"Precision @ Top 1%:     {prec_at_1:.4f}")
    print(f"Operational Lift:       {prec_at_1 / y_te_purch.mean():.2f}x")
    print("-"*45)
    print("Confusion Matrix (Purchase class):")
    print(cm)
    print("="*45)

if __name__ == "__main__":
    evaluate_minority_class()
