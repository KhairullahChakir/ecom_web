"""
Phase 3: CPU Latency Benchmarking
=================================
Measure inference latency for all models on CPU.
Report mean, p95, p99 latency from 1000 inferences.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import joblib
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "online_shoppers_intention.csv")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "models")
REPORTS_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "metrics")

# Force CPU
torch.set_num_threads(1)  # Single thread for consistent benchmarking


class MLPBlock(torch.nn.Module):
    """Single MLP block with batch norm and dropout"""
    def __init__(self, in_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()
        
    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.fc2(x)))
        return x


class TabMModel(torch.nn.Module):
    """TabM model for loading"""
    def __init__(self, n_features, n_classes=2, hidden_dim=128, n_ensemble=8, dropout=0.1):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.input_bn = torch.nn.BatchNorm1d(n_features)
        self.ensemble_blocks = torch.nn.ModuleList([
            MLPBlock(n_features, hidden_dim, dropout) for _ in range(n_ensemble)
        ])
        self.output_heads = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, 1) for _ in range(n_ensemble)
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


def prepare_sample_data():
    """Prepare sample data for benchmarking"""
    print("\n[1/4] Preparing sample data...")
    
    df = pd.read_csv(DATA_PATH)
    X = df.drop('Revenue', axis=1)
    
    # Encode categorical columns
    categorical_cols = ['Month', 'VisitorType', 'Weekend']
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Pick random samples for benchmarking
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_scaled), size=100, replace=False)
    samples = X_scaled[sample_indices]
    
    print(f"  → Prepared {len(samples)} samples for benchmarking")
    return samples, X.columns.tolist()


def load_models():
    """Load all trained models"""
    print("\n[2/4] Loading models...")
    
    models = {}
    
    # Logistic Regression
    lr_path = os.path.join(MODELS_PATH, 'logistic_regression.joblib')
    if os.path.exists(lr_path):
        models['Logistic Regression'] = joblib.load(lr_path)
        print("  → Loaded Logistic Regression")
    
    # XGBoost
    xgb_path = os.path.join(MODELS_PATH, 'xgboost.joblib')
    if os.path.exists(xgb_path):
        models['XGBoost'] = joblib.load(xgb_path)
        print("  → Loaded XGBoost")
    
    # TabM
    tabm_path = os.path.join(MODELS_PATH, 'tabm_model.pt')
    if os.path.exists(tabm_path):
        checkpoint = torch.load(tabm_path, map_location='cpu')
        config = checkpoint['config']
        n_features = checkpoint['n_features']
        
        tabm_model = TabMModel(
            n_features=n_features,
            hidden_dim=config['hidden_dim'],
            n_ensemble=config['n_ensemble'],
            dropout=config['dropout']
        )
        tabm_model.load_state_dict(checkpoint['model_state_dict'])
        tabm_model.eval()
        models['TabM'] = tabm_model
        print("  → Loaded TabM")
    
    return models


def benchmark_model(model, samples, model_name, n_iterations=1000, n_warmup=100):
    """Benchmark a single model"""
    latencies = []
    
    # Determine if PyTorch model
    is_pytorch = isinstance(model, torch.nn.Module)
    
    if is_pytorch:
        samples_tensor = torch.FloatTensor(samples)
        model.eval()
    
    # Warmup runs
    for i in range(n_warmup):
        sample = samples[i % len(samples)].reshape(1, -1)
        if is_pytorch:
            with torch.no_grad():
                _ = model(torch.FloatTensor(sample))
        else:
            _ = model.predict_proba(sample)
    
    # Benchmark runs
    for i in range(n_iterations):
        sample = samples[i % len(samples)].reshape(1, -1)
        
        start = time.perf_counter()
        if is_pytorch:
            with torch.no_grad():
                _ = model(torch.FloatTensor(sample))
        else:
            _ = model.predict_proba(sample)
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        'model': model_name,
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies))
    }


def run_benchmarks(models, samples, n_iterations=1000):
    """Run benchmarks for all models"""
    print(f"\n[3/4] Running benchmarks ({n_iterations} iterations each)...")
    
    results = []
    for name, model in models.items():
        print(f"  → Benchmarking {name}...", end=" ", flush=True)
        result = benchmark_model(model, samples, name, n_iterations)
        results.append(result)
        print(f"mean={result['mean_ms']:.3f}ms, p95={result['p95_ms']:.3f}ms")
    
    return results


def save_results(results):
    """Save benchmark results"""
    print("\n[4/4] Saving results...")
    
    os.makedirs(REPORTS_PATH, exist_ok=True)
    
    # Save latency report
    report = {
        'benchmark_config': {
            'n_iterations': 1000,
            'n_warmup': 100,
            'cpu_threads': 1,
            'single_sample_inference': True
        },
        'results': results
    }
    
    report_path = os.path.join(REPORTS_PATH, 'latency_benchmark.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  → Report saved to: {report_path}")
    
    # Print results table
    print("\n" + "=" * 80)
    print("LATENCY BENCHMARK RESULTS (milliseconds)")
    print("=" * 80)
    print(f"\n{'Model':<25} {'Mean':>10} {'Std':>10} {'P50':>10} {'P95':>10} {'P99':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<25} {r['mean_ms']:>10.3f} {r['std_ms']:>10.3f} {r['p50_ms']:>10.3f} {r['p95_ms']:>10.3f} {r['p99_ms']:>10.3f}")
    
    # Check target
    target_ms = 10.0
    print(f"\n  🎯 Target: <{target_ms}ms")
    for r in results:
        status = "✅" if r['p95_ms'] < target_ms else "⚠️"
        print(f"  {status} {r['model']}: p95={r['p95_ms']:.3f}ms {'(PASS)' if r['p95_ms'] < target_ms else '(NEEDS OPTIMIZATION)'}")
    
    return report


def main():
    """Main benchmark pipeline"""
    print("=" * 60)
    print("PHASE 3: CPU LATENCY BENCHMARKING")
    print("=" * 60)
    
    # Prepare data
    samples, feature_names = prepare_sample_data()
    
    # Load models
    models = load_models()
    
    if not models:
        print("\n❌ No models found! Please run training scripts first.")
        return
    
    # Run benchmarks
    results = run_benchmarks(models, samples)
    
    # Save results
    report = save_results(results)
    
    print("\n" + "=" * 60)
    print("✅ PHASE 3 COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  → Phase 4: Pruning experiments (reduce TabM ensemble size)")
    print("  → Phase 5: ONNX export for faster inference")
    
    return report


if __name__ == "__main__":
    main()
