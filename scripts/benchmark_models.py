"""
Export TCN to ONNX and Benchmark Both Models
"""

import torch
import numpy as np
import time
import onnxruntime as ort
from train_tcn import AbandonmentTCN

# Load TCN
print("Loading TCN model...")
tcn_model = AbandonmentTCN(
    num_page_types=8,
    embedding_dim=32,
    num_channels=[64, 64, 64],
    kernel_size=3,
    dropout=0.2
)
tcn_model.load_state_dict(torch.load('data/abandonment_tcn.pth', weights_only=True))
tcn_model.eval()

# Dummy inputs
batch_size = 1
seq_len = 20
dummy_page = torch.randint(0, 8, (batch_size, seq_len))
dummy_dur = torch.randn(batch_size, seq_len)

# Export TCN to ONNX
print("Exporting TCN to ONNX...")
with torch.no_grad():
    torch.onnx.export(
        tcn_model,
        (dummy_page, dummy_dur),
        "data/abandonment_tcn.onnx",
        input_names=['page_ids', 'durations'],
        output_names=['logits'],
        opset_version=14,
        export_params=True,
        do_constant_folding=True
    )
print("✅ TCN exported to data/abandonment_tcn.onnx")

# ============================
# Benchmark Both Models
# ============================

print("\n" + "="*50)
print("BENCHMARKING: Transformer vs TCN")
print("="*50)

# Test data
test_page = np.array([[1, 2, 3, 2, 4, 5, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
test_dur = np.array([[15.0, 30.0, 45.0, 20.0, 60.0, 10.0, 25.0, 35.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32) / 180.0

# Load both ONNX models
transformer_session = ort.InferenceSession("data/abandonment_transformer.onnx")
tcn_session = ort.InferenceSession("data/abandonment_tcn.onnx")

# Benchmark Transformer
trans_latencies = []
for _ in range(100):
    start = time.perf_counter()
    transformer_session.run(None, {'page_ids': test_page, 'durations': test_dur})
    trans_latencies.append((time.perf_counter() - start) * 1000)

# Benchmark TCN
tcn_latencies = []
for _ in range(100):
    start = time.perf_counter()
    tcn_session.run(None, {'page_ids': test_page, 'durations': test_dur})
    tcn_latencies.append((time.perf_counter() - start) * 1000)

# Results
trans_result = transformer_session.run(None, {'page_ids': test_page, 'durations': test_dur})
tcn_result = tcn_session.run(None, {'page_ids': test_page, 'durations': test_dur})

trans_prob = 1 / (1 + np.exp(-trans_result[0][0][0]))
tcn_prob = 1 / (1 + np.exp(-tcn_result[0][0][0]))

print(f"\n{'Model':<15} {'Accuracy':<12} {'Latency':<12} {'Prediction':<12}")
print("-" * 50)
print(f"{'Transformer':<15} {'86.7%':<12} {np.mean(trans_latencies):.3f}ms      {trans_prob*100:.1f}%")
print(f"{'TCN':<15} {'86.4%':<12} {np.mean(tcn_latencies):.3f}ms      {tcn_prob*100:.1f}%")

# Winner
print("\n" + "="*50)
if np.mean(tcn_latencies) < np.mean(trans_latencies):
    speedup = np.mean(trans_latencies) / np.mean(tcn_latencies)
    print(f"🏆 TCN is {speedup:.1f}x FASTER than Transformer!")
else:
    speedup = np.mean(tcn_latencies) / np.mean(trans_latencies)
    print(f"🏆 Transformer is {speedup:.1f}x FASTER than TCN!")
print("="*50)
