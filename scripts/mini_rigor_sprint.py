import torch
import torch.nn as nn
import numpy as np
import os
import sys
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.utils.data import DataLoader

# Add scripts dir for architecture
sys.path.append(os.path.abspath("scripts"))
from train_tcn import AbandonmentTCN, ClickstreamDataset

# Paths
DATA_DIR = "d:/op_ecom/data/processed_rigorous"
DEVICE = torch.device("cpu") # Use CPU for mini sprint stability

def load_subset(name, limit=50000):
    xp = np.load(os.path.join(DATA_DIR, f"X_page_{name}.npy"))[:limit]
    xd = np.load(os.path.join(DATA_DIR, f"X_dur_{name}.npy"))[:limit]
    y = np.load(os.path.join(DATA_DIR, f"y_{name}.npy"))[:limit]
    return xp, xd, y

def mini_rigor_sprint():
    print("🚀 Running Mini Rigor Sprint (50k samples)...")
    Xp_tr, Xd_tr, y_tr = load_subset("train", 50000)
    Xp_te, Xd_te, y_te = load_subset("test", 10000)
    
    train_loader = DataLoader(ClickstreamDataset(Xp_tr, Xd_tr, y_tr), batch_size=512, shuffle=True)
    
    model = AbandonmentTCN(num_page_types=4).to(DEVICE)
    # y=0 is purchase, y=1 is abandon. Weight class 0 (purchase)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 2 Epochs only
    for epoch in range(2):
        model.train()
        for p, d, label in train_loader:
            optimizer.zero_grad()
            out = model(p, d).squeeze()
            loss = criterion(out, label.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete")
        
    model.eval()
    with torch.no_grad():
        p_te = torch.from_numpy(Xp_te).long()
        d_te = torch.from_numpy(Xd_te).float()
        probs = torch.sigmoid(model(p_te, d_te).squeeze()).numpy()
    
    # y=1 is abandon, y=0 is purchase. 
    # For AUC (purchase detect), we want proba(success). success=1.
    y_te_success = 1 - y_te
    probs_success = 1 - probs # If proba(abandon) is low, proba(purchase) is high
    
    res_auc = roc_auc_score(y_te_success, probs_success)
    print(f"\n--- Mini-Rigor Baseline (Clean Data) ---")
    print(f"Revised ROC-AUC: {res_auc:.4f}")

if __name__ == "__main__":
    mini_rigor_sprint()
