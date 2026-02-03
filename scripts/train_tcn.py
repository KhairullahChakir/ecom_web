"""
TCN (Temporal Convolutional Network) for Abandonment Prediction
Compares against the Transformer model for clickstream analysis.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ============================
# TCN Architecture Components
# ============================

class CausalConv1d(nn.Module):
    """Causal convolution that only looks at past timesteps."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        out = self.conv(x)
        # Remove the future timesteps (causal)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    """Single TCN residual block with dilated causal convolutions."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        
        # Residual
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class AbandonmentTCN(nn.Module):
    """
    TCN for predicting session abandonment from clickstream sequences.
    
    Input:
        - page_ids: (batch, seq_len) - Page type indices
        - durations: (batch, seq_len) - Normalized duration values
    
    Output:
        - logits: (batch, 1) - Abandonment probability (before sigmoid)
    """
    
    def __init__(
        self,
        num_page_types: int = 8,
        embedding_dim: int = 32,
        num_channels: list = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Embeddings
        self.page_embedding = nn.Embedding(num_page_types, embedding_dim, padding_idx=0)
        self.duration_proj = nn.Linear(1, embedding_dim)
        self.combine = nn.Linear(embedding_dim * 2, num_channels[0])
        
        # TCN layers with exponentially increasing dilation
        layers = []
        in_channels = num_channels[0]
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i  # 1, 2, 4, 8...
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))
            in_channels = out_channels
        
        self.tcn = nn.Sequential(*layers)
        
        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, 1)
        )
    
    def forward(self, page_ids, durations):
        # Create embeddings
        page_emb = self.page_embedding(page_ids)  # (batch, seq, emb)
        dur_emb = self.duration_proj(durations.unsqueeze(-1))  # (batch, seq, emb)
        
        # Combine
        combined = torch.cat([page_emb, dur_emb], dim=-1)  # (batch, seq, emb*2)
        x = self.combine(combined)  # (batch, seq, channels)
        
        # TCN expects (batch, channels, seq)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        
        # Global average pooling over time
        x = x.mean(dim=2)  # (batch, channels)
        
        # Classify
        logits = self.classifier(x)
        return logits


# ============================
# Dataset (reuse from Transformer)
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
            torch.save(model.state_dict(), 'data/abandonment_tcn.pth')
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")
    
    print(f"\nBest Validation Accuracy: {best_val_acc*100:.1f}%")
    return history, best_val_acc


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
    
    # Create TCN model
    model = AbandonmentTCN(
        num_page_types=8,
        embedding_dim=32,
        num_channels=[64, 64, 64],  # 3 layers
        kernel_size=3,
        dropout=0.2
    )
    
    print(f"TCN Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    
    history, best_acc = train_model(model, train_loader, val_loader, epochs=30, device=device)
    
    print("\n✅ TCN Training Complete! Model saved to data/abandonment_tcn.pth")
    print(f"\n📊 COMPARISON:")
    print(f"   Transformer: 86.7% accuracy")
    print(f"   TCN:         {best_acc*100:.1f}% accuracy")
