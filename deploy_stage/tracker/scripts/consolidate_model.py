import onnx
import os

model_path = r"d:\op_ecom\tracker\models\tcn_real.onnx"
output_path = r"d:\op_ecom\tracker\models\tcn_real_standalone.onnx"

print(f"Loading model from {model_path}...")
model = onnx.load(model_path)

print(f"Saving standalone model to {output_path}...")
# By default, onnx.save embeds data if save_as_external_data is False
onnx.save(model, output_path, save_as_external_data=False)

print("Self-contained model created successfully.")
print(f"Size of new model: {os.path.getsize(output_path) / 1024:.2f} KB")
