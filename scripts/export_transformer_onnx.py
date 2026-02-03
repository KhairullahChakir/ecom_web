"""
Export Trained Abandonment Transformer to ONNX
"""

import torch
import numpy as np
from train_transformer import AbandonmentTransformer

# Load model
model = AbandonmentTransformer(
    num_page_types=8,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=128,
    max_seq_len=20,
    dropout=0.1
)

model.load_state_dict(torch.load('data/abandonment_transformer.pth', weights_only=True))
model.eval()

# Dummy inputs for tracing
batch_size = 1
seq_len = 20

dummy_page = torch.randint(0, 8, (batch_size, seq_len))
dummy_dur = torch.randn(batch_size, seq_len)

# Export to ONNX using legacy exporter (more compatible)
print("Exporting to ONNX...")

# Disable dynamic shapes for simpler export
with torch.no_grad():
    torch.onnx.export(
        model,
        (dummy_page, dummy_dur),
        "data/abandonment_transformer.onnx",
        input_names=['page_ids', 'durations'],
        output_names=['logits'],
        opset_version=14,  # Use older opset for better compatibility
        export_params=True,
        do_constant_folding=True
    )

print("✅ Exported to data/abandonment_transformer.onnx")

# Verify with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession("data/abandonment_transformer.onnx")

# Test inference
test_page = np.array([[1, 2, 3, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
test_dur = np.array([[15.0, 30.0, 45.0, 20.0, 60.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32) / 180.0

result = session.run(None, {'page_ids': test_page, 'durations': test_dur})
prob = 1 / (1 + np.exp(-result[0][0][0]))  # Sigmoid

print(f"Test Inference - Abandonment Probability: {prob*100:.1f}%")

# Benchmark latency
import time

latencies = []
for _ in range(100):
    start = time.perf_counter()
    session.run(None, {'page_ids': test_page, 'durations': test_dur})
    latencies.append((time.perf_counter() - start) * 1000)

avg_latency = np.mean(latencies)
print(f"Average Inference Latency: {avg_latency:.3f} ms")
print("\n🚀 Transformer ONNX Ready for Production!")
