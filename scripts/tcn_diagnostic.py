"""
TCN Model Overfitting Diagnostic (Robust Loading)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import os

# ============================
# TCN Architecture Components
# ============================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation)
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class AbandonmentTCN(nn.Module):
    def __init__(self, num_page_types: int = 8, embedding_dim: int = 32, num_channels: list = [64, 64, 64], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.page_embedding = nn.Embedding(num_page_types, embedding_dim, padding_idx=0)
        self.duration_proj = nn.Linear(1, embedding_dim)
        self.combine = nn.Linear(embedding_dim * 2, num_channels[0])
        layers = []
        in_channels = num_channels[0]
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, 1)
        )
    def forward(self, page_ids, durations):
        page_emb = self.page_embedding(page_ids)
        dur_emb = self.duration_proj(durations.unsqueeze(-1))
        combined = torch.cat([page_emb, dur_emb], dim=-1)
        x = self.combine(combined)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.mean(dim=2)
        logits = self.classifier(x)
        return logits

def run_diagnostic():
    print("=" * 60)
    print("TCN MODEL DIAGNOSTIC")
    print("=" * 60)
    
    # 1. Inspect Checkpoint
    model_path = "scripts/data/tcn_real.pth"
    print(f"\n🔍 Inspecting checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    for key, val in checkpoint.items():
        if 'weight' in key or 'bias' in key:
            print(f"   {key:<30} : {list(val.shape)}")
    
    # Infer parameters from checkpoint
    # page_embedding.weight shape should be [num_page_types, embedding_dim]
    emb_shape = checkpoint['page_embedding.weight'].shape
    inferred_num_page_types = emb_shape[0]
    inferred_embedding_dim = emb_shape[1]
    
    # num_channels from classifier or TCN blocks
    # TCN blocks: tcn.0.conv1.conv.weight shape is [out_channels, in_channels, kernel_size]
    # Actually simpler: count TCN blocks
    num_tcn_layers = sum(1 for key in checkpoint.keys() if 'tcn' in key and 'conv1.conv.weight' in key)
    print(f"\n💡 Inferred Parameters:")
    print(f"   num_page_types: {inferred_num_page_types}")
    print(f"   embedding_dim:  {inferred_embedding_dim}")
    print(f"   TCN Layers:     {num_tcn_layers}")
    
    # 2. Instantiate Model
    print("\n🤖 Instantiating model...")
    model = AbandonmentTCN(
        num_page_types=inferred_num_page_types,
        embedding_dim=inferred_embedding_dim,
        num_channels=[64] * num_tcn_layers
    )
    
    model.load_state_dict(checkpoint)
    model.eval()
    print("   ✅ Successfully loaded!")
    
    # 3. Load Data
    print("\n📊 Loading data...")
    X_page = np.load("scripts/data/X_page_real.npy")
    X_dur = np.load("scripts/data/X_dur_real.npy")
    y = np.load("scripts/data/y_abandon_real.npy")
    
    # Split: 80% train, 10% val, 10% test
    print("\n📂 Splitting data (80% train, 10% val, 10% test)...")
    # First split: 80% train, 20% temp
    X_p_tr, X_p_temp, X_d_tr, X_d_temp, y_tr, y_temp = train_test_split(
        X_page, X_dur, y, test_size=0.2, random_state=42, stratify=y
    )
    # Second split: 50% of temp (which is 10% of total) into val and te
    X_p_va, X_p_te, X_d_va, X_d_te, y_va, y_te = train_test_split(
        X_p_temp, X_d_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"   Train samples: {len(y_tr):,}")
    print(f"   Val samples:   {len(y_va):,}")
    print(f"   Test samples:  {len(y_te):,}")
    
    # 4. Evaluate
    def get_metrics(X_p, X_d, y_target, name):
        print(f"\n🔍 {name}:")
        X_p_t = torch.from_numpy(X_p).long()
        X_d_t = torch.from_numpy(X_d).float()
        
        preds = []
        batch_size = 2000
        with torch.no_grad():
            for i in range(0, len(X_p), batch_size):
                out = model(X_p_t[i:i+batch_size], X_d_t[i:i+batch_size]).squeeze(-1)
                preds.extend((torch.sigmoid(out) > 0.5).float().numpy())
        
        preds = np.array(preds)
        acc = accuracy_score(y_target, preds)
        print(f"   Accuracy: {acc*100:.2f}%")
        return preds, acc

    tr_preds, tr_acc = get_metrics(X_p_tr, X_d_tr, y_tr, "TRAIN")
    va_preds, va_acc = get_metrics(X_p_va, X_d_va, y_va, "VALIDATION")
    te_preds, te_acc = get_metrics(X_p_te, X_d_te, y_te, "TEST")
    
    # 5. Overfitting & Confusion Matrix
    print("\n" + "=" * 60)
    print("🔬 RESULTS")
    print("=" * 60)
    print(f"Gap: {(tr_acc - te_acc)*100:.2f}%")
    
    print("\nConfusion Matrix (Test):")
    cm = confusion_matrix(y_te, te_preds)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_te, te_preds, target_names=['No Abandon', 'Abandon']))
    
    baseline = max((y == 0).mean(), (y == 1).mean())
    print(f"Baseline: {baseline*100:.1f}%")

if __name__ == "__main__":
    run_diagnostic()
