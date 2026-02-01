import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import time
import os
import onnxruntime as ort

# Force CPU for fair benchmark if CUDA not present
DEVICE = torch.device("cpu")
print(f"🔥 Using Device: {DEVICE}")

DATA_PATH = "d:/op_ecom/data/raw/online_shoppers_intention.csv"
MODELS_PATH = "d:/op_ecom/backend/models"
os.makedirs(MODELS_PATH, exist_ok=True)

# 1. Data Loading
def load_and_preprocess():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        # Fallback for relative path if needed
        return None
    
    df = pd.read_csv(DATA_PATH)
    y = df['Revenue'].astype(int).values
    cat_cols = ['Month', 'VisitorType', 'Weekend', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
    num_cols = [c for c in df.columns if c not in cat_cols and c != 'Revenue']
    
    X_cat = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    cat_dims = []
    for i, col in enumerate(cat_cols):
        le = LabelEncoder()
        X_cat[:, i] = le.fit_transform(df[col].astype(str))
        cat_dims.append(len(le.classes_))
        
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols].astype(float))
    return X_cat, X_num, y, cat_dims, X_num.shape[1]

# 2. Model
class FeatureEmbedding(nn.Module):
    def __init__(self, cat_dims, embed_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(d, embed_dim) for d in cat_dims
        ])
    def forward(self, x_cat):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(embedded, dim=1)

class TabM(nn.Module):
    def __init__(self, cat_dims, num_dim, hidden_dim=128, n_ensemble=4):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.embedding = FeatureEmbedding(cat_dims, embed_dim=4)
        input_dim = (len(cat_dims) * 4) + num_dim
        self.bn_in = nn.BatchNorm1d(input_dim)
        self.ensemble_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            ) for _ in range(n_ensemble)
        ])
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_ensemble)
        ])
        
    def forward(self, x_cat, x_num):
        emb_cat = self.embedding(x_cat).flatten(1)
        x_in = torch.cat([emb_cat, x_num], dim=1)
        x_in = self.bn_in(x_in)
        outputs = [head(block(x_in)) for block, head in zip(self.ensemble_blocks, self.heads)]
        stacked = torch.stack(outputs, dim=0)
        return stacked.mean(dim=0)

# 3. Train
def train_model(model, full_data, name, epochs=10, batch_size=256):
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    dataset = TensorDataset(*full_data['train'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_auc = 0
    print(f"Training {name} (K={model.n_ensemble})...")
    
    for epoch in range(epochs):
        model.train()
        for xc, xn, y in loader:
            xc, xn, y = xc.to(DEVICE), xn.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(xc, xn).squeeze()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
        # Quick Val
        model.eval()
        with torch.no_grad():
            xc_v, xn_v, y_v = [t.to(DEVICE) for t in full_data['val']]
            val_out = model(xc_v, xn_v).squeeze()
            val_probs = torch.sigmoid(val_out).cpu().numpy()
            try:
                val_auc = roc_auc_score(y_v.cpu().numpy(), val_probs)
            except:
                val_auc = 0.5
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"{MODELS_PATH}/{name}.pth")
            
    print(f"  Best AUC: {best_auc:.4f}")
    return best_auc

# 4. Benchmark ONNX
def benchmark_onnx(model_name, n_ensemble, cat_dims, num_dim):
    # Load and export
    model = TabM(cat_dims, num_dim, n_ensemble=n_ensemble)
    try:
        model.load_state_dict(torch.load(f"{MODELS_PATH}/{model_name}.pth"))
    except:
        print(f"  Could not load {model_name}")
        return 999.0
        
    model.eval()
    dummy_cat = torch.zeros((1, len(cat_dims)), dtype=torch.long)
    dummy_num = torch.randn((1, num_dim), dtype=torch.float32)
    onnx_path = f"{MODELS_PATH}/{model_name}.onnx"
    
    torch.onnx.export(
        model, (dummy_cat, dummy_num), onnx_path,
        input_names=['input_cat', 'input_num'], output_names=['output'],
        dynamic_axes={'input_cat': {0: 'batch'}, 'input_num': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    
    # Run
    sess = ort.InferenceSession(onnx_path)
    inp = {'input_cat': dummy_cat.numpy(), 'input_num': dummy_num.numpy()}
    
    latencies = []
    for _ in range(50): sess.run(None, inp) # Warmup
    for _ in range(500):
        start = time.perf_counter()
        sess.run(None, inp)
        latencies.append((time.perf_counter() - start) * 1000)
    
    return np.mean(latencies)

def main():
    X_c, X_n, y, cat_dims, num_dim = load_and_preprocess()
    
    indices = np.arange(len(y))
    # Stratified split to ensure we have classes
    Xc_tr, Xc_temp, Xn_tr, Xn_temp, y_tr, y_temp = train_test_split(X_c, X_n, y, test_size=0.2, stratify=y, random_state=42)
    Xc_val, Xc_test, Xn_val, Xn_test, y_val, y_test = train_test_split(Xc_temp, Xn_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    def to_tensor(xc, xn, y_):
        return torch.tensor(xc, dtype=torch.long), torch.tensor(xn, dtype=torch.float32), torch.tensor(y_, dtype=torch.float32)

    full_data = {
        'train': to_tensor(Xc_tr, Xn_tr, y_tr),
        'val': to_tensor(Xc_val, Xn_val, y_val)
    }
    
    print("\n--- Starting Pruning Experiments ---")
    
    # 1. Train K=4
    auc_k4 = train_model(TabM(cat_dims, num_dim, n_ensemble=4), full_data, "tabm_k4_pruned", epochs=5)
    lat_k4 = benchmark_onnx("tabm_k4_pruned", 4, cat_dims, num_dim)
    
    # 2. Train K=2
    auc_k2 = train_model(TabM(cat_dims, num_dim, n_ensemble=2), full_data, "tabm_k2_pruned", epochs=5)
    lat_k2 = benchmark_onnx("tabm_k2_pruned", 2, cat_dims, num_dim)
    
    # 3. Train K=1
    auc_k1 = train_model(TabM(cat_dims, num_dim, n_ensemble=1), full_data, "tabm_k1_pruned", epochs=5)
    lat_k1 = benchmark_onnx("tabm_k1_pruned", 1, cat_dims, num_dim)
    
    print("\n=== FINAL RESULTS ===")
    print(f"K=4 | AUC: {auc_k4:.4f} | Latency: {lat_k4:.4f} ms")
    print(f"K=2 | AUC: {auc_k2:.4f} | Latency: {lat_k2:.4f} ms")
    print(f"K=1 | AUC: {auc_k1:.4f} | Latency: {lat_k1:.4f} ms")

if __name__ == "__main__":
    main()
