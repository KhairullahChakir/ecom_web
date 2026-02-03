"""
Abandonment Prediction Transformer
A lightweight Transformer Encoder for predicting user session abandonment.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math

# ============================
# Model Architecture
# ============================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence order awareness."""
    
    def __init__(self, d_model: int, max_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AbandonmentTransformer(nn.Module):
    """
    Lightweight Transformer for predicting session abandonment.
    
    Input:
        - page_ids: (batch, seq_len) - Page type indices
        - durations: (batch, seq_len) - Normalized duration values
    
    Output:
        - logits: (batch, 1) - Abandonment probability (before sigmoid)
    """
    
    def __init__(
        self,
        num_page_types: int = 8,  # 7 types + 1 padding
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        max_seq_len: int = 20,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Embeddings
        self.page_embedding = nn.Embedding(num_page_types, d_model, padding_idx=0)
        self.duration_proj = nn.Linear(1, d_model)
        self.combine = nn.Linear(d_model * 2, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, page_ids, durations):
        # Create embeddings
        page_emb = self.page_embedding(page_ids)  # (batch, seq, d_model)
        dur_emb = self.duration_proj(durations.unsqueeze(-1))  # (batch, seq, d_model)
        
        # Combine page and duration information
        combined = torch.cat([page_emb, dur_emb], dim=-1)  # (batch, seq, d_model*2)
        x = self.combine(combined)  # (batch, seq, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask (where page_id == 0)
        padding_mask = (page_ids == 0)  # (batch, seq)
        
        # Transformer forward
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Global average pooling (ignoring padding)
        mask = (~padding_mask).float().unsqueeze(-1)  # (batch, seq, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (batch, d_model)
        
        # Classify
        logits = self.classifier(x)
        
        return logits


# ============================
# Dataset
# ============================

class ClickstreamDataset(Dataset):
    def __init__(self, X_page, X_dur, y):
        self.X_page = torch.from_numpy(X_page).long()
        self.X_dur = torch.from_numpy(X_dur).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X_page[idx], self.X_dur[idx], self.y[idx]


# ============================
# Training Loop
# ============================

def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
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
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for page, dur, label in val_loader:
                page, dur, label = page.to(device), dur.to(device), label.to(device)
                
                logits = model(page, dur).squeeze(-1)
                loss = criterion(logits, label)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == label).sum().item()
                total += label.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'data/abandonment_transformer.pth')
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")
    
    print(f"\nBest Validation Accuracy: {best_val_acc*100:.1f}%")
    return history


if __name__ == "__main__":
    # Load data
    X_page = np.load("data/X_page.npy")
    X_dur = np.load("data/X_dur.npy")
    y = np.load("data/y_abandon.npy")
    
    # Train/Val split
    split_idx = int(len(y) * 0.8)
    train_dataset = ClickstreamDataset(X_page[:split_idx], X_dur[:split_idx], y[:split_idx])
    val_dataset = ClickstreamDataset(X_page[split_idx:], X_dur[split_idx:], y[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Create model
    model = AbandonmentTransformer(
        num_page_types=8,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=20,
        dropout=0.1
    )
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    
    history = train_model(model, train_loader, val_loader, epochs=30, device=device)
    
    print("\n✅ Training Complete! Model saved to data/abandonment_transformer.pth")
