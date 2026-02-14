import json

notebook_path = r'd:\op_ecom\notebooks\03_latency_benchmarking.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
torchscript_added = False

for cell in nb.get('cells', []):
    # Detect the ONNX section and insert TorchScript before it
    if cell.get('cell_type') == 'markdown' and '4. ONNX Optimization' in cell.get('source', [[""]])[0]:
        # Add TorchScript Markdown
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. TorchScript (JIT Traced) Optimization\n",
                "\n",
                "TorchScript is the built-in PyTorch optimization. We use **tracing** to record the model's operations for a fixed input shape. This often improves performance on CPUs."
            ]
        })
        # Add TorchScript Code
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if 'm_tabm' in locals():\n",
                "    print(\"\\n⚡ Benchmarking TorchScript (JIT Trace)...\")\n",
                "    \n",
                "    # Create traced model\n",
                "    with torch.no_grad():\n",
                "        traced_model = torch.jit.trace(m_tabm, (sample_cat, sample_num))\n",
                "    \n",
                "    # Benchmark function\n",
                "    def benchmark_jit(model, x_c, x_n, iters=500):\n",
                "        with torch.no_grad():\n",
                "            for _ in range(50): _ = model(x_c, x_n)\n",
                "            \n",
                "        latencies = []\n",
                "        for _ in range(iters):\n",
                "            start = time.perf_counter()\n",
                "            with torch.no_grad(): _ = model(x_c, x_n)\n",
                "            latencies.append((time.perf_counter() - start) * 1000)\n",
                "        return np.mean(latencies)\n",
                "\n",
                "    ts_lat = benchmark_jit(traced_model, sample_cat, sample_num)\n",
                "    results.append({'Model': 'TabM (TorchScript)', 'Type': 'Neural (Traced)', 'Latency (ms)': ts_lat})\n",
                "    print(f\"TabM (TorchScript): {ts_lat:.3f} ms\")"
            ]
        })
        torchscript_added = True

    # Update numbering in existing markdown headers if needed
    if torchscript_added:
        if cell.get('cell_type') == 'markdown':
            source = cell.get('source', [])
            if source and source[0].startswith('## 4. ONNX'):
                cell['source'] = [source[0].replace('4.', '5.')]
            elif source and source[0].startswith('## 5. Final'):
                cell['source'] = [source[0].replace('5.', '6.')]

    new_cells.append(cell)

nb['cells'] = new_cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched with TorchScript section.")
