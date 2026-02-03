"""
Export real-data models to ONNX
"""
import torch
import numpy as np
import onnxruntime as ort
from train_transformer import AbandonmentTransformer
from train_tcn import AbandonmentTCN
from train_gru import AbandonmentGRU

# Dummy inputs
batch_size = 1
seq_len = 20
dummy_page = torch.randint(0, 4, (batch_size, seq_len))
dummy_dur = torch.randn(batch_size, seq_len)

# 1. Transformer
temp_model = AbandonmentTransformer(num_page_types=4)
temp_model.load_state_dict(torch.load('data/transformer_real.pth', weights_only=True))
temp_model.eval()
torch.onnx.export(temp_model, (dummy_page, dummy_dur), "data/transformer_real.onnx", input_names=['page_ids', 'durations'], output_names=['logits'], opset_version=14)
print("✅ Transformer exported")

# 2. TCN
temp_model = AbandonmentTCN(num_page_types=4)
temp_model.load_state_dict(torch.load('data/tcn_real.pth', weights_only=True))
temp_model.eval()
torch.onnx.export(temp_model, (dummy_page, dummy_dur), "data/tcn_real.onnx", input_names=['page_ids', 'durations'], output_names=['logits'], opset_version=14)
print("✅ TCN exported")

# 3. GRU
temp_model = AbandonmentGRU(num_page_types=4)
temp_model.load_state_dict(torch.load('data/gru_real.pth', weights_only=True))
temp_model.eval()
torch.onnx.export(temp_model, (dummy_page, dummy_dur), "data/gru_real.onnx", input_names=['page_ids', 'durations'], output_names=['logits'], opset_version=14)
print("✅ GRU exported")
