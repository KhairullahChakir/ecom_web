"""
Phase 4: Pruning / Shrinking Experiments
=========================================
Run TabM with different ensemble sizes to find accuracy vs latency trade-off.
Ensemble sizes: 16 → 12 → 8 → 4 → 2
Generate trade-off plot.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "online_shoppers_intention.csv")
REPORTS_PATH = os.path.join(os.path.dirname(__file__), "..", "reports")
FIGURES_PATH = os.path.join(REPORTS_PATH, "figures")

# Force CPU with single thread
torch.set_num_threads(1)
DEVICE = torch.device("cpu")


class MLPBlock(nn.Module):
    """Single MLP block"""
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
    """TabM with configurable ensemble size"""
    def __init__(self, n_features, hidden_dim=128, n_ensemble=8, dropout=0.1):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.input_bn = nn.BatchNorm1d(n_features)
        self.ensemble_blocks = nn.ModuleList([
            MLPBlock(n_features, hidden_dim, dropout) for _ in range(n_ensemble)
        ])
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_ensemble)
        ])
        
    def forward(self, x):
        x = self.input_bn(x)
        outputs = []
        for block, head in zip(self.ensemble_blocks, self.output_heads):
            h = block(x)
            out = head(h)
            outputs.append(out)
        stacked = torch.stack(outputs, dim=0)
        mean_output = stacked.mean(dim=0)
        return mean_output.squeeze(-1)


def load_data():
    """Load and preprocess data"""
    df = pd.read_csv(DATA_PATH)
    X = df.drop('Revenue', axis=1)
    y = df['Revenue'].astype(int)
    
    categorical_cols = ['Month', 'VisitorType', 'Weekend']
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': torch.FloatTensor(X_train_scaled),
        'X_val': torch.FloatTensor(X_val_scaled),
        'X_test': torch.FloatTensor(X_test_scaled),
        'y_train': torch.FloatTensor(y_train.values),
        'y_val': torch.FloatTensor(y_val.values),
        'y_test': torch.FloatTensor(y_test.values),
        'y_test_np': y_test.values,
        'n_features': X_train_scaled.shape[1]
    }


def train_model(data, n_ensemble, hidden_dim=128, epochs=50, patience=10):
    """Train TabM with specific ensemble size"""
    model = TabMModel(
        n_features=data['n_features'],
        hidden_dim=hidden_dim,
        n_ensemble=n_ensemble,
        dropout=0.15
    ).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    best_val_auc = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(data['X_val'])
            val_probs = torch.sigmoid(val_outputs).numpy()
            val_auc = roc_auc_score(data['y_val'].numpy(), val_probs)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    model.load_state_dict(best_state)
    return model


def evaluate_model(model, data):
    """Evaluate model on test set"""
    model.eval()
    with torch.no_grad():
        test_outputs = model(data['X_test'])
        test_probs = torch.sigmoid(test_outputs).numpy()
        test_preds = (test_probs >= 0.5).astype(int)
    
    return {
        'auc_roc': float(roc_auc_score(data['y_test_np'], test_probs)),
        'f1': float(f1_score(data['y_test_np'], test_preds))
    }


def benchmark_latency(model, sample, n_iterations=500):
    """Benchmark inference latency"""
    model.eval()
    sample_tensor = torch.FloatTensor(sample.reshape(1, -1))
    
    # Warmup
    for _ in range(50):
        with torch.no_grad():
            _ = model(sample_tensor)
    
    # Benchmark
    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(sample_tensor)
        latencies.append((time.perf_counter() - start) * 1000)
    
    return {
        'mean_ms': float(np.mean(latencies)),
        'p95_ms': float(np.percentile(latencies, 95))
    }


def create_tradeoff_plot(results):
    """Create accuracy vs latency trade-off plot"""
    os.makedirs(FIGURES_PATH, exist_ok=True)
    
    ensembles = [r['n_ensemble'] for r in results]
    aucs = [r['auc_roc'] for r in results]
    latencies = [r['p95_ms'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Primary axis: AUC
    color1 = '#1E4FA8'  # لاجوردی
    ax1.set_xlabel('Ensemble Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUC-ROC', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(ensembles, aucs, 'o-', color=color1, linewidth=2, markersize=10, label='AUC-ROC')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.85, 0.95)
    
    # Secondary axis: Latency
    ax2 = ax1.twinx()
    color2 = '#E74C3C'
    ax2.set_ylabel('P95 Latency (ms)', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(ensembles, latencies, 's--', color=color2, linewidth=2, markersize=10, label='P95 Latency')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Target line
    ax2.axhline(y=10, color='green', linestyle=':', linewidth=2, alpha=0.7, label='10ms Target')
    
    # Annotations
    for i, (ens, auc, lat) in enumerate(zip(ensembles, aucs, latencies)):
        ax1.annotate(f'{auc:.3f}', (ens, auc), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9, color=color1)
        ax2.annotate(f'{lat:.2f}ms', (ens, lat), textcoords="offset points", 
                    xytext=(0, -15), ha='center', fontsize=9, color=color2)
    
    # Title and legend
    plt.title('TabM Pruning: Accuracy vs Latency Trade-off', fontsize=14, fontweight='bold', pad=20)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right')
    
    plt.tight_layout()
    plot_path = os.path.join(FIGURES_PATH, 'pruning_tradeoff.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def main():
    """Main pruning experiment pipeline"""
    print("=" * 60)
    print("PHASE 4: PRUNING / SHRINKING EXPERIMENTS")
    print("=" * 60)
    
    # Ensemble sizes to test
    ensemble_sizes = [16, 12, 8, 4, 2]
    
    print("\n[1/4] Loading data...")
    data = load_data()
    print(f"  → Features: {data['n_features']}, Samples: {len(data['X_train'])}")
    
    # Get sample for latency benchmark
    sample = data['X_test'][0].numpy()
    
    print(f"\n[2/4] Running pruning sweep ({ensemble_sizes})...")
    results = []
    
    for n_ens in ensemble_sizes:
        print(f"\n  → Training TabM (ensemble={n_ens})...", end=" ", flush=True)
        
        # Train
        model = train_model(data, n_ens)
        
        # Evaluate
        metrics = evaluate_model(model, data)
        
        # Benchmark
        latency = benchmark_latency(model, sample)
        
        result = {
            'n_ensemble': n_ens,
            'auc_roc': metrics['auc_roc'],
            'f1': metrics['f1'],
            'mean_ms': latency['mean_ms'],
            'p95_ms': latency['p95_ms']
        }
        results.append(result)
        
        print(f"AUC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}, p95={latency['p95_ms']:.2f}ms")
    
    print("\n[3/4] Generating trade-off plot...")
    plot_path = create_tradeoff_plot(results)
    print(f"  → Plot saved: {plot_path}")
    
    print("\n[4/4] Saving results...")
    os.makedirs(os.path.join(REPORTS_PATH, "metrics"), exist_ok=True)
    report = {
        'experiment': 'TabM Pruning Sweep',
        'ensemble_sizes': ensemble_sizes,
        'results': results
    }
    report_path = os.path.join(REPORTS_PATH, "metrics", "pruning_results.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  → Results saved: {report_path}")
    
    # Print results table
    print("\n" + "=" * 70)
    print("PRUNING RESULTS")
    print("=" * 70)
    print(f"\n{'Ensemble':>10} {'AUC-ROC':>12} {'F1':>12} {'Mean (ms)':>12} {'P95 (ms)':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['n_ensemble']:>10} {r['auc_roc']:>12.4f} {r['f1']:>12.4f} {r['mean_ms']:>12.3f} {r['p95_ms']:>12.3f}")
    
    # Find best trade-off
    best = max([r for r in results if r['p95_ms'] < 10], key=lambda x: x['auc_roc'])
    print(f"\n  🏆 Best trade-off: ensemble={best['n_ensemble']} (AUC={best['auc_roc']:.4f}, p95={best['p95_ms']:.2f}ms)")
    
    print("\n" + "=" * 60)
    print("✅ PHASE 4 COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  → Phase 5: Export best model to ONNX")
    print("  → Phase 6: Integrate into FastAPI")
    
    return results


if __name__ == "__main__":
    main()
