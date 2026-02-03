"""
Final Benchmark: Transformer vs TCN vs GRU
"""

import torch
import numpy as np
import time
import onnxruntime as ort
from train_gru import AbandonmentGRU

# Load and export GRU
print("Loading GRU model...")
gru_model = AbandonmentGRU(
    num_page_types=8,
    embedding_dim=32,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    bidirectional=True
)
gru_model.load_state_dict(torch.load('data/abandonment_gru.pth', weights_only=True))
gru_model.eval()

# Dummy inputs
batch_size = 1
seq_len = 20
dummy_page = torch.randint(0, 8, (batch_size, seq_len))
dummy_dur = torch.randn(batch_size, seq_len)

# Export GRU to ONNX
print("Exporting GRU to ONNX...")
with torch.no_grad():
    torch.onnx.export(
        gru_model,
        (dummy_page, dummy_dur),
        "data/abandonment_gru.onnx",
        input_names=['page_ids', 'durations'],
        output_names=['logits'],
        opset_version=14,
        export_params=True,
        do_constant_folding=True
    )
print("✅ GRU exported to data/abandonment_gru.onnx")

# ============================
# Benchmark All Three Models
# ============================

print("\n" + "="*60)
print("       FINAL BENCHMARK: Transformer vs TCN vs GRU")
print("="*60)

# Test data
test_page = np.array([[1, 2, 3, 2, 4, 5, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
test_dur = np.array([[15.0, 30.0, 45.0, 20.0, 60.0, 10.0, 25.0, 35.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32) / 180.0

# Load all ONNX models
transformer_session = ort.InferenceSession("data/abandonment_transformer.onnx")
tcn_session = ort.InferenceSession("data/abandonment_tcn.onnx")
gru_session = ort.InferenceSession("data/abandonment_gru.onnx")

def benchmark(session, name, n_runs=100):
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {'page_ids': test_page, 'durations': test_dur})
        latencies.append((time.perf_counter() - start) * 1000)
    
    result = session.run(None, {'page_ids': test_page, 'durations': test_dur})
    prob = 1 / (1 + np.exp(-result[0].flatten()[0]))
    
    return np.mean(latencies), np.std(latencies), prob

# Run benchmarks
trans_lat, trans_std, trans_prob = benchmark(transformer_session, "Transformer")
tcn_lat, tcn_std, tcn_prob = benchmark(tcn_session, "TCN")
gru_lat, gru_std, gru_prob = benchmark(gru_session, "GRU")

# Results table
print(f"\n{'Model':<15} {'Accuracy':<12} {'Latency':<15} {'Prediction':<12} {'Params':<12}")
print("-" * 65)
print(f"{'Transformer':<15} {'86.7%':<12} {trans_lat:.3f} ± {trans_std:.3f}ms   {trans_prob*100:.1f}%        {'78K':<12}")
print(f"{'TCN':<15} {'86.4%':<12} {tcn_lat:.3f} ± {tcn_std:.3f}ms   {tcn_prob*100:.1f}%        {'81K':<12}")
print(f"{'GRU':<15} {'86.6%':<12} {gru_lat:.3f} ± {gru_std:.3f}ms   {gru_prob*100:.1f}%        {'137K':<12}")

# Find winner
print("\n" + "="*60)
print("                        WINNER ANALYSIS")
print("="*60)

latencies = {'Transformer': trans_lat, 'TCN': tcn_lat, 'GRU': gru_lat}
accuracies = {'Transformer': 86.7, 'TCN': 86.4, 'GRU': 86.6}

fastest = min(latencies, key=latencies.get)
most_accurate = max(accuracies, key=accuracies.get)

print(f"\n🏆 FASTEST:       {fastest} ({latencies[fastest]:.3f}ms)")
print(f"🎯 MOST ACCURATE: {most_accurate} ({accuracies[most_accurate]}%)")

# Overall recommendation
print("\n" + "-"*60)
print("RECOMMENDATION:")
if fastest == most_accurate:
    print(f"   ✅ {fastest} is the clear winner (fastest AND most accurate)")
else:
    print(f"   ⚡ For SPEED:    Use {fastest}")
    print(f"   🎯 For ACCURACY: Use {most_accurate}")
    print(f"   ⚖️  BALANCED:    Use TCN (good speed + competitive accuracy)")
print("-"*60)
