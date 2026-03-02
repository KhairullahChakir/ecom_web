import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from torch.utils.data import DataLoader

# Add scripts dir for architecture
sys.path.append(os.path.abspath("scripts"))
from train_tcn import AbandonmentTCN, ClickstreamDataset

DATA_DIR = "d:/op_ecom/data/processed_rigorous"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_eval_rigorous(X_tr_p, X_tr_d, y_tr, X_te_p, X_te_d, y_te, seed):
    print(f"\n--- Running Seed {seed} ---")
    torch.manual_seed(seed)
    
    train_loader = DataLoader(ClickstreamDataset(X_tr_p, X_tr_d, y_tr), batch_size=4096, shuffle=True)
    
    model = AbandonmentTCN(num_page_types=4).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 3 Epochs
    model.train()
    for epoch in range(3):
        total_loss = 0
        for p, d, label in train_loader:
            p, d, label = p.to(DEVICE), d.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            out = model(p, d).squeeze()
            loss = criterion(out, label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
    
    model.eval()
    all_probs = []
    with torch.no_grad():
        batch_size = 4096
        for i in range(0, len(X_te_p), batch_size):
            p = torch.from_numpy(X_te_p[i:i+batch_size]).long().to(DEVICE)
            d = torch.from_numpy(X_te_d[i:i+batch_size]).float().to(DEVICE)
            probs = torch.sigmoid(model(p, d).squeeze())
            all_probs.extend(probs.cpu().numpy())
    
    y_probs_purch = 1 - np.array(all_probs)
    y_te_purch = (y_te == 0).astype(int)
    
    prec, rec, _ = precision_recall_curve(y_te_purch, y_probs_purch)
    
    return {
        'ROC-AUC': roc_auc_score(y_te_purch, y_probs_purch),
        'PR-AUC': auc(rec, prec)
    }

def main():
    print(f"🚀 RIGOROUS STABILITY TEST RUNNING ON: {DEVICE}")
    print("Loading rigorous data...")
    X_tr_p = np.load(os.path.join(DATA_DIR, "X_page_train.npy"))
    X_tr_d = np.load(os.path.join(DATA_DIR, "X_dur_train.npy"))
    y_tr = np.load(os.path.join(DATA_DIR, "y_train.npy"))

    X_te_p = np.load(os.path.join(DATA_DIR, "X_page_test.npy"))
    X_te_d = np.load(os.path.join(DATA_DIR, "X_dur_test.npy"))
    y_te = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    seeds = [42, 7, 123, 999, 2026]
    results = []

    start_time = time.time()
    for s in seeds:
        res = train_eval_rigorous(X_tr_p, X_tr_d, y_tr, X_te_p, X_te_d, y_te, s)
        results.append(res)
        print(f"   Results: ROC-AUC={res['ROC-AUC']:.4f}, PR-AUC={res['PR-AUC']:.4f}")

    end_time = time.time()
    print(f"\n✅ Final Rigorous Test Complete in {(end_time - start_time)/60:.2f} minutes.")

    df_res = pd.DataFrame(results)
    summary = {
        'Metric': ['Buyer ROC-AUC (Rigorous)', 'Buyer PR-AUC (Rigorous)'],
        'Mean': [df_res['ROC-AUC'].mean(), df_res['PR-AUC'].mean()],
        'Std Dev (σ)': [df_res['ROC-AUC'].std(), df_res['PR-AUC'].std()]
    }
    print("\n========================================")
    print("  FINAL RIGOROUS PERFORMANCE SUMMARY")
    print("========================================")
    print(pd.DataFrame(summary).to_string(index=False))
    print("========================================")

if __name__ == "__main__":
    main()
