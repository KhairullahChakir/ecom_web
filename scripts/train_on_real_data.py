"""
Train all 3 models on REAL RetailRocket data.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import time

# Import model architectures
from train_transformer import AbandonmentTransformer, ClickstreamDataset
from train_tcn import AbandonmentTCN
from train_gru import AbandonmentGRU

# Load REAL data
print("Loading REAL RetailRocket data...")
X_page = np.load("data/X_page_real.npy")
X_dur = np.load("data/X_dur_real.npy")
y = np.load("data/y_abandon_real.npy")

print(f"Data shape: {X_page.shape}")
print(f"Class distribution: {y.mean()*100:.1f}% abandoned")

# Subsample for faster training (use 50K samples)
np.random.seed(42)
indices = np.random.permutation(len(y))[:50000]
X_page = X_page[indices]
X_dur = X_dur[indices]
y = y[indices]

print(f"Using {len(y):,} samples for training")

# Train/Val split
split_idx = int(len(y) * 0.8)
train_dataset = ClickstreamDataset(X_page[:split_idx], X_dur[:split_idx], y[:split_idx])
val_dataset = ClickstreamDataset(X_page[split_idx:], X_dur[split_idx:], y[split_idx:])

# Handle class imbalance with weighted sampler
train_labels = y[:split_idx]
class_counts = np.bincount(train_labels.astype(int))
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_labels.astype(int)]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=128)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device}")

def train_model(model, name, epochs=20):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for page, dur, label in train_loader:
            page, dur, label = page.to(device), dur.to(device), label.to(device)
            
            optimizer.zero_grad()
            logits = model(page, dur).squeeze(-1)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for page, dur, label in val_loader:
                page, dur, label = page.to(device), dur.to(device), label.to(device)
                logits = model(page, dur).squeeze(-1)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == label).sum().item()
                total += label.size(0)
        
        val_acc = correct / total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'data/{name}_real.pth')
        
        print(f"[{name}] Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.1f}%")
    
    return best_val_acc

# Train all 3 models
results = {}

# 1. Transformer
print("\n" + "="*50)
print("Training TRANSFORMER on REAL data...")
print("="*50)
transformer = AbandonmentTransformer(num_page_types=4)  # 3 event types + padding
results['Transformer'] = train_model(transformer, 'transformer', epochs=20)

# 2. TCN
print("\n" + "="*50)
print("Training TCN on REAL data...")
print("="*50)
tcn = AbandonmentTCN(num_page_types=4)
results['TCN'] = train_model(tcn, 'tcn', epochs=20)

# 3. GRU
print("\n" + "="*50)
print("Training GRU on REAL data...")
print("="*50)
gru = AbandonmentGRU(num_page_types=4)
results['GRU'] = train_model(gru, 'gru', epochs=20)

# Summary
print("\n" + "="*60)
print("       RESULTS ON REAL RETAILROCKET DATA")
print("="*60)
print(f"\n{'Model':<15} {'Accuracy':<12} {'vs Paper (74.3%)':<20}")
print("-" * 50)
for name, acc in results.items():
    diff = acc * 100 - 74.3
    sign = "+" if diff > 0 else ""
    print(f"{name:<15} {acc*100:.1f}%        {sign}{diff:.1f}%")

print("\n" + "="*60)
