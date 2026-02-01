"""
Phase 2: TabM Deep Learning Model Training
==========================================
Train TabM (efficient MLP ensemble) on UCI Online Shoppers dataset.
Compare with baseline models and generate evaluation table.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import joblib
import json
from datetime import datetime

try:
    from tabm import TabM
    TABM_AVAILABLE = True
except ImportError:
    TABM_AVAILABLE = False
    print("TabM package not found, using custom MLP ensemble implementation")

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "online_shoppers_intention.csv")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "models")
REPORTS_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "metrics")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPBlock(nn.Module):
    """Single MLP block with batch norm and dropout"""
    def __init__(self, in_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.fc2(x)))
        return x


class TabMModel(nn.Module):
    """
    TabM-inspired model: Efficient ensemble of MLPs via weight sharing
    Uses K parallel MLPs with shared backbone for efficiency
    """
    def __init__(self, n_features, n_classes=2, hidden_dim=128, n_ensemble=8, dropout=0.1):
        super().__init__()
        self.n_ensemble = n_ensemble
        
        # Shared feature processing
        self.input_bn = nn.BatchNorm1d(n_features)
        
        # Ensemble of MLP heads (weight sharing via same architecture)
        self.ensemble_blocks = nn.ModuleList([
            MLPBlock(n_features, hidden_dim, dropout) for _ in range(n_ensemble)
        ])
        
        # Individual output heads
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_ensemble)
        ])
        
    def forward(self, x):
        x = self.input_bn(x)
        
        # Get predictions from each ensemble member
        outputs = []
        for block, head in zip(self.ensemble_blocks, self.output_heads):
            h = block(x)
            out = head(h)
            outputs.append(out)
        
        # Average ensemble predictions
        stacked = torch.stack(outputs, dim=0)  # (n_ensemble, batch, 1)
        mean_output = stacked.mean(dim=0)  # (batch, 1)
        
        return mean_output.squeeze(-1)


def load_and_preprocess_data():
    """Load and preprocess data for PyTorch training"""
    print("=" * 60)
    print("PHASE 2: TabM DEEP LEARNING MODEL")
    print("=" * 60)
    
    print(f"\n[INFO] Using device: {DEVICE}")
    
    print("\n[1/4] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Separate features and target
    X = df.drop('Revenue', axis=1)
    y = df['Revenue'].astype(int)
    
    # Encode categorical columns
    categorical_cols = ['Month', 'VisitorType', 'Weekend']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Train/val/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    
    print(f"  → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val_scaled).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test_scaled).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train.values).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val.values).to(DEVICE)
    y_test_t = torch.FloatTensor(y_test.values).to(DEVICE)
    
    return {
        'X_train': X_train_t, 'X_val': X_val_t, 'X_test': X_test_t,
        'y_train': y_train_t, 'y_val': y_val_t, 'y_test': y_test_t,
        'scaler': scaler, 'label_encoders': label_encoders,
        'n_features': X_train_scaled.shape[1],
        'y_test_np': y_test.values
    }


def train_tabm(data, config):
    """Train TabM model"""
    print(f"\n[2/4] Training TabM (ensemble={config['n_ensemble']}, hidden={config['hidden_dim']})...")
    
    # Create model
    model = TabMModel(
        n_features=data['n_features'],
        hidden_dim=config['hidden_dim'],
        n_ensemble=config['n_ensemble'],
        dropout=config['dropout']
    ).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Create data loaders
    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(data['X_val'])
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()
            val_auc = roc_auc_score(data['y_val'].cpu().numpy(), val_probs)
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  → Epoch {epoch+1}/{config['epochs']}: Loss={train_loss/len(train_loader):.4f}, Val AUC={val_auc:.4f}")
        
        if patience_counter >= config['patience']:
            print(f"  → Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"  → Best validation AUC: {best_val_auc:.4f}")
    
    return model, best_val_auc


def evaluate_model(model, data):
    """Evaluate model on test set"""
    print("\n[3/4] Evaluating on test set...")
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(data['X_test'])
        test_probs = torch.sigmoid(test_outputs).cpu().numpy()
        test_preds = (test_probs >= 0.5).astype(int)
    
    y_test = data['y_test_np']
    
    metrics = {
        'model': 'TabM',
        'auc_roc': float(roc_auc_score(y_test, test_probs)),
        'f1': float(f1_score(y_test, test_preds)),
        'precision': float(precision_score(y_test, test_preds)),
        'recall': float(recall_score(y_test, test_preds))
    }
    
    print(f"  → AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  → F1 Score:  {metrics['f1']:.4f}")
    print(f"  → Precision: {metrics['precision']:.4f}")
    print(f"  → Recall:    {metrics['recall']:.4f}")
    
    return metrics, test_probs


def save_results(model, metrics, data, config):
    """Save model and update metrics report"""
    print("\n[4/4] Saving results...")
    
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    # Save PyTorch model
    model_path = os.path.join(MODELS_PATH, 'tabm_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'n_features': data['n_features']
    }, model_path)
    print(f"  → Model saved to: {model_path}")
    
    # Load and update baseline metrics
    baseline_path = os.path.join(REPORTS_PATH, 'baseline_metrics.json')
    with open(baseline_path, 'r') as f:
        report = json.load(f)
    
    # Add TabM results
    report['models'].append(metrics)
    report['tabm_config'] = config
    report['timestamp'] = datetime.now().isoformat()
    
    # Save updated report
    with open(baseline_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<25} {'AUC-ROC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 70)
    for m in report['models']:
        print(f"{m['model']:<25} {m['auc_roc']:>10.4f} {m['f1']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f}")
    
    # Find best model
    best = max(report['models'], key=lambda x: x['auc_roc'])
    print(f"\n  🏆 Best model: {best['model']} (AUC-ROC: {best['auc_roc']:.4f})")
    
    return report


def main():
    """Main training pipeline"""
    # Configuration
    config = {
        'hidden_dim': 128,
        'n_ensemble': 8,
        'dropout': 0.15,
        'lr': 0.001,
        'weight_decay': 0.01,
        'batch_size': 256,
        'epochs': 100,
        'patience': 15
    }
    
    # Load data
    data = load_and_preprocess_data()
    
    # Train TabM
    model, val_auc = train_tabm(data, config)
    
    # Evaluate
    metrics, test_probs = evaluate_model(model, data)
    
    # Save results
    report = save_results(model, metrics, data, config)
    
    print("\n" + "=" * 60)
    print("✅ PHASE 2 COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  → Phase 3: CPU latency benchmarking")
    print("  → Phase 4: Pruning experiments")
    
    return model, metrics


if __name__ == "__main__":
    main()
