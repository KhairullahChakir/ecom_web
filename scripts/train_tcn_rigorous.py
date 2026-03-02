import torch
import torch.nn as nn
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Add scripts dir for architecture
sys.path.append(os.path.abspath("scripts"))
from train_tcn import AbandonmentTCN, ClickstreamDataset

# Paths
DATA_DIR = "d:/op_ecom/data/processed_rigorous"
MODEL_PATH = "backend/models/exit_model_tcn_rigorous.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_rigorous_loader(name, batch_size=4096, shuffle=False):
    xp = np.load(os.path.join(DATA_DIR, f"X_page_{name}.npy"))
    xd = np.load(os.path.join(DATA_DIR, f"X_dur_{name}.npy"))
    y = np.load(os.path.join(DATA_DIR, f"y_{name}.npy"))
    dataset = ClickstreamDataset(xp, xd, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_rigorous():
    print(f"🚀 Training on {DEVICE}...")
    train_loader = load_rigorous_loader("train", shuffle=True)
    val_loader = load_rigorous_loader("val", shuffle=False)
    
    model = AbandonmentTCN(num_page_types=4).to(DEVICE)
    # Using pos_weight based on the 0.77% rate for buyers (Class 0)
    # y=0 is Purchase, y=1 is Abandon. We want to weight buyers.
    # Standard BCEWithLogitsLoss weights the positive class (y=1).
    # Since y=1 is the majority, we want to weight y=0 more.
    # Simplest: use 1.0 weight and use a custom loss or flip labels.
    # Let's keep labels same and use pos_weight to penalize missing y=1 (Abandons).
    # Wait, if we want to find buyers (y=0), we should probably use weights.
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for p, d, label in train_loader:
            p, d, label = p.to(DEVICE), d.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            out = model(p, d).squeeze()
            loss = criterion(out, label.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for p, d, label in val_loader:
                p, d, label = p.to(DEVICE), d.to(DEVICE), label.to(DEVICE)
                out = model(p, d).squeeze()
                val_loss += criterion(out, label.float()).item()
        
        avg_tr = train_loss / len(train_loader)
        avg_vl = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train: {avg_tr:.4f} | Val: {avg_vl:.4f}")
        
        if avg_vl < best_val_loss:
            best_val_loss = avg_vl
            torch.save(model.state_dict(), MODEL_PATH)
            print("⭐ Saved Best Model")

if __name__ == "__main__":
    train_rigorous()
