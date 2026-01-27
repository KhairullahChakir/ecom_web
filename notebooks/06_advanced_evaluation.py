"""
Phase 9: Advanced Evaluation & Thesis-Grade Metrics
===================================================
Implementation of high-impact metrics for professional portfolios:
1. PR-AUC (Precision-Recall Area Under Curve)
2. Brier Score (Calibration/Reliability)
3. Precision@TopK (Top 5%, 10% conversion quality)
4. Latency Breakdown (Preprocess vs Inference vs Total)
"""

import os
import time
import json
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from sklearn.metrics import (
    precision_recall_curve, auc, brier_score_loss, 
    roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "online_shoppers_intention.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "models")
ONNX_PATH = os.path.join(MODELS_DIR, "tabm_best.onnx")
SCALER_PATH = os.path.join(MODELS_DIR, "onnx_scaler.joblib")
ENCODERS_PATH = os.path.join(MODELS_DIR, "onnx_label_encoders.joblib")
REPORTS_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "metrics")

def get_precision_at_k(y_true, y_probs, k=0.1):
    """Calculate precision at top K percent of probabilities"""
    n = len(y_probs)
    k_idx = int(n * k)
    # Sort indices by probability in descending order
    top_indices = np.argsort(y_probs)[::-1][:k_idx]
    # Calculate precision in that top K
    return np.mean(y_true[top_indices])

def load_data():
    """Consistent 70/10/20 split as used in Phase 2"""
    df = pd.read_csv(DATA_PATH)
    X = df.drop('Revenue', axis=1)
    y = df['Revenue'].astype(int).values
    
    # Pre-split to get the exact test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_test, y_test

def run_advanced_evaluation():
    print("=" * 60)
    print("PHASE 9: ADVANCED METRICS & LATENCY BREAKDOWN")
    print("=" * 60)
    
    # 1. Load Artifacts
    print("\n[1/3] Loading models and test data...")
    X_test_df, y_test = load_data()
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    session = ort.InferenceSession(ONNX_PATH, sess_opts, providers=['CPUExecutionProvider'])
    
    # 2. Advanced Metrics
    print("\n[2/3] Calculating Decision Quality metrics...")
    
    all_probs = []
    prep_latencies = []
    inf_latencies = []
    
    categorical_cols = ['Month', 'VisitorType', 'Weekend']
    
    # Evaluate sample by sample to get accurate latency distribution
    for i in range(len(X_test_df)):
        sample = X_test_df.iloc[i].copy()
        
        # --- PREPROCESS ---
        t0 = time.perf_counter()
        # Categorical encoding
        for col in categorical_cols:
            le = label_encoders[col]
            try:
                sample[col] = le.transform([str(sample[col])])[0]
            except:
                sample[col] = 0
        
        # Scaling
        sample_scaled = scaler.transform(sample.values.reshape(1, -1)).astype(np.float32)
        prep_latencies.append((time.perf_counter() - t0) * 1000)
        
        # --- INFERENCE ---
        t1 = time.perf_counter()
        input_name = session.get_inputs()[0].name
        onnx_result = session.run(None, {input_name: sample_scaled})
        # The output of tabm onnx is usually [1, 1] or [1]
        prob = float(onnx_result[0].flatten()[0])
        inf_latencies.append((time.perf_counter() - t1) * 1000)
        
        all_probs.append(prob)

    all_probs = np.array(all_probs)
    y_preds = (all_probs >= 0.5).astype(int)
    
    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, all_probs)
    pr_auc = auc(recall, precision)
    
    # Brier Score (lower is better, calibration)
    brier = brier_score_loss(y_test, all_probs)
    
    # Precision@TopK
    p_at_5 = get_precision_at_k(y_test, all_probs, k=0.05)
    p_at_10 = get_precision_at_k(y_test, all_probs, k=0.10)
    
    print(f"  → PR-AUC:      {pr_auc:.4f} (Focus on conversion recall)")
    print(f"  → Brier Score:  {brier:.4f} (Model reliability)")
    print(f"  → Precision@5%: {p_at_5:.4f} (Quality of top leads)")
    print(f"  → Precision@10%:{p_at_10:.4f}")

    # 3. Latency Breakdown
    print("\n[3/3] Analyzing Latency Components (ms)...")
    latency_stats = {
        "preprocessing": {
            "mean": np.mean(prep_latencies),
            "p95": np.percentile(prep_latencies, 95)
        },
        "inference": {
            "mean": np.mean(inf_latencies),
            "p95": np.percentile(inf_latencies, 95)
        }
    }
    
    print(f"  → Preprocessing: {latency_stats['preprocessing']['mean']:.3f}ms (p95: {latency_stats['preprocessing']['p95']:.3f}ms)")
    print(f"  → Inference:     {latency_stats['inference']['mean']:.3f}ms (p95: {latency_stats['inference']['p95']:.3f}ms)")

    # Save to file
    final_report = {
        "model_name": "TabM (Ensemble=4)",
        "split_protocol": "70/10/20 Stratified, Seed 42",
        "advanced_metrics": {
            "roc_auc": roc_auc_score(y_test, all_probs),
            "pr_auc": pr_auc,
            "brier_score": brier,
            "precision_at_5pct": p_at_5,
            "precision_at_10pct": p_at_10,
            "f1": f1_score(y_test, y_preds)
        },
        "latency_stats": latency_stats,
        "environment": {
            "engine": "ONNX Runtime v1.16+",
            "threads_intra": 1,
            "split": "Train: 6904 | Val: 986 | Test: 2466"
        }
    }
    
    os.makedirs(REPORTS_PATH, exist_ok=True)
    report_path = os.path.join(REPORTS_PATH, "advanced_evaluation.json")
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n✅ Advanced report saved to: {report_path}")

if __name__ == "__main__":
    run_advanced_evaluation()
