import torch
import torch.nn as nn
import numpy as np
import os
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

# Add scripts dir to sys path for architecture imports
sys.path.append(os.path.abspath("scripts"))
from train_tcn import AbandonmentTCN
from train_transformer import ClickstreamDataset

# Config
DATA_DIR = "data/processed"
MODEL_SAVE_PATH = "backend/models/exit_model_tcn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 1024
LR = 1e-3

def train():
    print(f"🚀 Training with DEVICE: {DEVICE}")
    
    # 1. Load Data
    X_page = np.load(os.path.join(DATA_DIR, "X_page_real.npy"))
    X_dur = np.load(os.path.join(DATA_DIR, "X_dur_real.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_abandon_real.npy"))
    
    # 2. Split (80/10/10 Stratified)
    train_idx, temp_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=y[temp_idx], random_state=42)
    
    train_loader = DataLoader(ClickstreamDataset(X_page[train_idx], X_dur[train_idx], y[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ClickstreamDataset(X_page[val_idx], X_dur[val_idx], y[val_idx]), batch_size=BATCH_SIZE)
    
    # 3. Model
    model = AbandonmentTCN(num_page_types=4).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    best_val_pr = 0
    patience = 5
    counter = 0
    
    print(f"📊 Dataset Size: {len(y)} | Train: {len(train_idx)} | Val: {len(val_idx)}")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for page, dur, label in train_loader:
            page, dur, label = page.to(DEVICE), dur.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            logits = model(page, dur).squeeze()
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for page, dur, label in val_loader:
                page, dur, label = page.to(DEVICE), dur.to(DEVICE), label.to(DEVICE)
                logits = model(page, dur).squeeze()
                probs = torch.sigmoid(logits)
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(label.cpu().numpy())
        
        val_pr = average_precision_score(val_labels, val_probs)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val PR-AUC: {val_pr:.4f}")
        
        if val_pr > best_val_pr:
            best_val_pr = val_pr
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✨ New Best Model Saved (PR-AUC: {best_val_pr:.4f})")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("🛑 Early stopping triggered.")
                break

if __name__ == "__main__":
    train()
