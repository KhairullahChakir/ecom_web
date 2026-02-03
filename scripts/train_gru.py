"""
GRU (Gated Recurrent Unit) for Abandonment Prediction
Compares against Transformer and TCN models.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ============================
# GRU Architecture
# ============================

class AbandonmentGRU(nn.Module):
    """
    Bidirectional GRU for predicting session abandonment.
    
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
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embeddings
        self.page_embedding = nn.Embedding(num_page_types, embedding_dim, padding_idx=0)
        self.duration_proj = nn.Linear(1, embedding_dim)
        self.input_proj = nn.Linear(embedding_dim * 2, hidden_size)
        
        # GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output head
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, page_ids, durations):
        batch_size = page_ids.size(0)
        
        # Create embeddings
        page_emb = self.page_embedding(page_ids)  # (batch, seq, emb)
        dur_emb = self.duration_proj(durations.unsqueeze(-1))  # (batch, seq, emb)
        
        # Combine and project
        combined = torch.cat([page_emb, dur_emb], dim=-1)  # (batch, seq, emb*2)
        x = self.input_proj(combined)  # (batch, seq, hidden)
        
        # GRU forward
        gru_out, hidden = self.gru(x)  # gru_out: (batch, seq, hidden*2)
        
        # Use the last hidden state
        # For bidirectional, concatenate forward and backward final states
        if self.bidirectional:
            # hidden shape: (num_layers*2, batch, hidden)
            forward_final = hidden[-2, :, :]  # (batch, hidden)
            backward_final = hidden[-1, :, :]  # (batch, hidden)
            final_hidden = torch.cat([forward_final, backward_final], dim=-1)  # (batch, hidden*2)
        else:
            final_hidden = hidden[-1, :, :]  # (batch, hidden)
        
        # Classify
        logits = self.classifier(final_hidden)
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
            torch.save(model.state_dict(), 'data/abandonment_gru.pth')
        
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
    
    # Create GRU model
    model = AbandonmentGRU(
        num_page_types=8,
        embedding_dim=32,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        bidirectional=True
    )
    
    print(f"GRU Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    
    history, best_acc = train_model(model, train_loader, val_loader, epochs=30, device=device)
    
    print("\n✅ GRU Training Complete! Model saved to data/abandonment_gru.pth")
    print(f"\n📊 COMPARISON SO FAR:")
    print(f"   Transformer: 86.7% accuracy")
    print(f"   TCN:         86.4% accuracy")
    print(f"   GRU:         {best_acc*100:.1f}% accuracy")
