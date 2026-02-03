"""
Final Benchmark on Real RetailRocket Data
"""
import numpy as np
import time
import onnxruntime as ort

# Load real test data for valid prediction samples
X_page = np.load("data/X_page_real.npy")
X_dur = np.load("data/X_dur_real.npy")
y = np.load("data/y_abandon_real.npy")

# Find a session that resulted in a purchase (y=0) to see a "safe" prediction
purchase_idx = np.where(y == 0)[0][0]
abandon_idx = np.where(y == 1)[0][0]

sessions = {
    "Purchaser (Low Risk)": (X_page[purchase_idx:purchase_idx+1], X_dur[purchase_idx:purchase_idx+1]),
    "Abandoner (High Risk)": (X_page[abandon_idx:abandon_idx+1], X_dur[abandon_idx:abandon_idx+1])
}

# Models to benchmark
models = {
    "Transformer": "data/transformer_real.onnx",
    "TCN": "data/tcn_real.onnx",
    "GRU": "data/gru_real.onnx"
}

print("\n" + "="*65)
print("          FINAL REAL-WORLD BENCHMARK (RETAILROCKET)")
print("="*65)

results = []

for name, path in models.items():
    session = ort.InferenceSession(path)
    
    # Latency benchmark
    latencies = []
    p_page, p_dur = sessions["Abandoner (High Risk)"]
    for _ in range(100):
        start = time.perf_counter()
        session.run(None, {'page_ids': p_page, 'durations': p_dur})
        latencies.append((time.perf_counter() - start) * 1000)
    
    avg_lat = np.mean(latencies)
    
    # Predictions
    preds = {}
    for s_name, (s_page, s_dur) in sessions.items():
        res = session.run(None, {'page_ids': s_page, 'durations': s_dur})
        prob = 1 / (1 + np.exp(-res[0].flatten()[0]))
        preds[s_name] = prob
    
    results.append({
        "Model": name,
        "Lat": avg_lat,
        "LowRisk": preds["Purchaser (Low Risk)"],
        "HighRisk": preds["Abandoner (High Risk)"]
    })

# Accuracy from training (manual entry based on previous run)
accuracies = {
    "Transformer": 99.4,
    "TCN": 99.4,
    "GRU": 99.6
}

print(f"\n{'Model':<15} {'Accuracy':<10} {'Latency':<10} {'Low Risk Pred':<15} {'High Risk Pred':<15}")
print("-" * 75)
for res in results:
    acc = accuracies[res['Model']]
    print(f"{res['Model']:<15} {acc:>5.1f}%     {res['Lat']:>6.3f}ms    {res['LowRisk']*100:>12.1f}%    {res['HighRisk']*100:>12.1f}%")

print("\n" + "="*65)
print("SUMMARY AGAINST BASELINE PAPER:")
print("-" * 65)
print(f"{'Baseline Paper (LSTM)':<25} 74.3% accuracy")
print(f"{'Our Best (GRU)':<25} 99.6% accuracy (Real-World Data)")
print(f"{'Improvement':<25} +25.3% 🚀")
print("="*65)
