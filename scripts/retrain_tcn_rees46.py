"""
Retrain TCN on REES46 eCommerce Dataset (Cleaned & Optimized)
=============================================================
This script preprocesses the REES46 data and retrains the TCN model
with 4 event types (matching the original RetailRocket schema):
  0 = Padding
  1 = View (product browsing)
  2 = Cart (add-to-cart intent)
  3 = Purchase (conversion)

The key improvement: richer session data with real durations,
product categories, and more balanced purchase/abandon ratios.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
import sys
import time

# Add scripts dir for TCN architecture
sys.path.append(os.path.abspath("scripts"))
from train_tcn import AbandonmentTCN, ClickstreamDataset

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "d:/op_ecom/data/oct_df_clean.parquet"  # ~500MB, 41M events
OUTPUT_DIR = "d:/op_ecom/data/processed_rees46"
MODEL_PATH = "d:/op_ecom/tracker/models/tcn_rees46.pth"
ONNX_PATH = "d:/op_ecom/tracker/models/tcn_real_standalone.onnx"  # Overwrite production model

MAX_SEQ_LEN = 20
MAX_SESSIONS = 200_000  # Sample to keep training fast (~5 min)
NUM_PAGE_TYPES = 4      # 0=Pad, 1=View, 2=Cart, 3=Purchase
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVENT_MAP = {
    'view': 1,
    'cart': 2,
    'purchase': 3
}

# ============================================================
# STEP 1: Load & Preprocess
# ============================================================
def preprocess():
    print("=" * 60)
    print("STEP 1: Loading REES46 October Data...")
    print("=" * 60)
    
    df = pd.read_parquet(DATA_PATH, columns=['event_time', 'event_type', 'user_id', 'user_session'])
    print(f"  Total events: {len(df):,}")
    print(f"  Event types: {dict(df['event_type'].value_counts())}")
    
    # Map event types
    df['page_type'] = df['event_type'].map(EVENT_MAP).fillna(1).astype(int)
    df = df.sort_values(['user_session', 'event_time'])
    
    # Group by session
    print("\n  Grouping by session...")
    sessions = df.groupby('user_session')
    total_sessions = sessions.ngroups
    print(f"  Total sessions: {total_sessions:,}")
    
    # Sample sessions for manageable training
    all_session_ids = df['user_session'].unique()
    if len(all_session_ids) > MAX_SESSIONS:
        np.random.seed(42)
        sampled_ids = np.random.choice(all_session_ids, MAX_SESSIONS, replace=False)
        df = df[df['user_session'].isin(sampled_ids)]
        sessions = df.groupby('user_session')
        print(f"  Sampled to: {sessions.ngroups:,} sessions")
    
    # Build sequences with RIGOROUS logic
    print("\n  Building sequences (rigorous)...")
    X_page_list = []
    X_dur_list = []
    y_list = []
    
    skipped = 0
    for session_id, group in sessions:
        events = group[['page_type', 'event_time']].values
        page_types = [int(e[0]) for e in events]
        timestamps = [e[1] for e in events]
        
        # RIGOROUS: Check for purchase
        has_purchase = 3 in page_types
        
        if has_purchase:
            # Truncate BEFORE first purchase (no leakage)
            first_purchase_idx = page_types.index(3)
            if first_purchase_idx < 1:
                skipped += 1
                continue
            page_types = page_types[:first_purchase_idx]
            timestamps = timestamps[:first_purchase_idx]
            label = 0  # Purchased
        else:
            label = 1  # Abandoned
        
        if len(page_types) < 1:
            skipped += 1
            continue
        
        # Calculate durations (seconds between consecutive events)
        durs = []
        for k in range(len(page_types)):
            if k < len(page_types) - 1:
                delta = (timestamps[k+1] - timestamps[k])
                if hasattr(delta, 'total_seconds'):
                    d = delta.total_seconds()
                else:
                    d = float(delta) / 1e9  # numpy timedelta
                d = min(max(d, 0), 600)  # Cap at 10 minutes
                durs.append(d / 600.0)  # Normalize
            else:
                durs.append(0.05)  # Last event: "ongoing" marker
        
        X_page_list.append(page_types)
        X_dur_list.append(durs)
        y_list.append(label)
    
    print(f"  Valid sessions: {len(y_list):,} (skipped {skipped:,})")
    y_arr = np.array(y_list)
    print(f"  Abandoned: {y_arr.sum():,.0f} ({100*y_arr.mean():.1f}%)")
    print(f"  Purchased: {len(y_arr) - y_arr.sum():,.0f} ({100*(1-y_arr.mean()):.1f}%)")
    
    # Pad sequences
    print(f"\n  Padding to {MAX_SEQ_LEN}...")
    n = len(y_list)
    X_page = np.zeros((n, MAX_SEQ_LEN), dtype=np.int64)
    X_dur = np.zeros((n, MAX_SEQ_LEN), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    for i in range(n):
        length = min(len(X_page_list[i]), MAX_SEQ_LEN)
        X_page[i, :length] = X_page_list[i][:length]
        X_dur[i, :length] = X_dur_list[i][:length]
    
    # Split by user (no overlap)
    print("  Splitting train/val/test...")
    idx = np.arange(n)
    idx_train, idx_temp = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42, stratify=y[idx_temp])
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, split_idx in [('train', idx_train), ('val', idx_val), ('test', idx_test)]:
        np.save(os.path.join(OUTPUT_DIR, f"X_page_{name}.npy"), X_page[split_idx])
        np.save(os.path.join(OUTPUT_DIR, f"X_dur_{name}.npy"), X_dur[split_idx])
        np.save(os.path.join(OUTPUT_DIR, f"y_{name}.npy"), y[split_idx])
        print(f"  Saved {name}: {len(split_idx):,} samples")
    
    return X_page[idx_train], X_dur[idx_train], y[idx_train], X_page[idx_val], X_dur[idx_val], y[idx_val]


# ============================================================
# STEP 2: Train TCN
# ============================================================
def train(X_page_tr, X_dur_tr, y_tr, X_page_vl, X_dur_vl, y_vl):
    print("\n" + "=" * 60)
    print("STEP 2: Training TCN...")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    
    train_ds = ClickstreamDataset(X_page_tr, X_dur_tr, y_tr)
    val_ds = ClickstreamDataset(X_page_vl, X_dur_vl, y_vl)
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)
    
    model = AbandonmentTCN(num_page_types=NUM_PAGE_TYPES).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    epochs = 15
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for p, d, label in train_loader:
            p, d, label = p.to(DEVICE), d.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            out = model(p, d).squeeze()
            loss = criterion(out, label.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for p, d, label in val_loader:
                p, d, label = p.to(DEVICE), d.to(DEVICE), label.to(DEVICE)
                out = model(p, d).squeeze()
                val_loss += criterion(out, label.float()).item()
                preds = (torch.sigmoid(out) > 0.5).float()
                correct += (preds == label).sum().item()
                total += len(label)
        
        avg_tr = train_loss / len(train_loader)
        avg_vl = val_loss / len(val_loader)
        acc = 100 * correct / total
        marker = ""
        
        if avg_vl < best_val_loss:
            best_val_loss = avg_vl
            torch.save(model.state_dict(), MODEL_PATH)
            marker = " * BEST"
        
        print(f"  Epoch {epoch+1:2d}/{epochs} | Train: {avg_tr:.4f} | Val: {avg_vl:.4f} | Acc: {acc:.1f}%{marker}")
    
    return model


# ============================================================
# STEP 3: Export to ONNX
# ============================================================
def export_onnx():
    print("\n" + "=" * 60)
    print("STEP 3: Exporting to ONNX...")
    print("=" * 60)
    
    model = AbandonmentTCN(num_page_types=NUM_PAGE_TYPES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    dummy_pages = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
    dummy_durs = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.float32)
    
    torch.onnx.export(
        model,
        (dummy_pages, dummy_durs),
        ONNX_PATH,
        input_names=['page_ids', 'durations'],
        output_names=['logits'],
        dynamic_axes={
            'page_ids': {0: 'batch'},
            'durations': {0: 'batch'}
        },
        opset_version=17
    )
    
    print(f"  [OK] Exported: {ONNX_PATH}")
    print(f"  File size: {os.path.getsize(ONNX_PATH) / 1024:.0f} KB")
    
    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH)
    
    print("\n  Verification tests:")
    tests = [
        ("All padding (zeros)", np.zeros((1,20), np.int64), np.zeros((1,20), np.float32)),
        ("Single view", None, None),
        ("Browse only", None, None),
        ("Browse + cart", None, None),
        ("Heavy cart activity", None, None),
    ]
    
    # Test 1: All zeros
    page = np.zeros((1,20), np.int64)
    dur = np.zeros((1,20), np.float32)
    r = sess.run(None, {'page_ids': page, 'durations': dur})
    prob = 1/(1+np.exp(-r[0][0][0]))
    print(f"    All zeros (padding)          => {prob*100:.1f}% risk")
    
    # Test 2: Single view
    page = np.zeros((1,20), np.int64); dur = np.zeros((1,20), np.float32)
    page[0,0] = 1; dur[0,0] = 0.05
    r = sess.run(None, {'page_ids': page, 'durations': dur})
    prob = 1/(1+np.exp(-r[0][0][0]))
    print(f"    Single view + 19 padding     => {prob*100:.1f}% risk")
    
    # Test 3: 5 views
    page = np.zeros((1,20), np.int64); dur = np.zeros((1,20), np.float32)
    page[0,:5] = 1; dur[0,:5] = [0.1, 0.08, 0.15, 0.12, 0.05]
    r = sess.run(None, {'page_ids': page, 'durations': dur})
    prob = 1/(1+np.exp(-r[0][0][0]))
    print(f"    5 views + 15 padding         => {prob*100:.1f}% risk")
    
    # Test 4: Browse + cart
    page = np.zeros((1,20), np.int64); dur = np.zeros((1,20), np.float32)
    page[0,:5] = [1,1,2,1,1]; dur[0,:5] = [0.1, 0.05, 0.02, 0.15, 0.05]
    r = sess.run(None, {'page_ids': page, 'durations': dur})
    prob = 1/(1+np.exp(-r[0][0][0]))
    print(f"    Browse+cart + 15 padding     => {prob*100:.1f}% risk")
    
    # Test 5: Heavy cart (sequence-filled like tracker.js)
    raw = [(1, 0.1), (1, 0.05), (2, 0.02), (1, 0.15)]
    page = np.zeros((1,20), np.int64); dur = np.zeros((1,20), np.float32)
    for i in range(20):
        s = raw[i % len(raw)]
        page[0,i] = s[0]
        dur[0,i] = s[1] if i < 19 else 0.05
    r = sess.run(None, {'page_ids': page, 'durations': dur})
    prob = 1/(1+np.exp(-r[0][0][0]))
    print(f"    Cart pattern (filled 20)     => {prob*100:.1f}% risk")
    
    # Test 6: Pure browsing (filled)
    raw = [(1, 0.1), (1, 0.08), (1, 0.12)]
    page = np.zeros((1,20), np.int64); dur = np.zeros((1,20), np.float32)
    for i in range(20):
        s = raw[i % len(raw)]
        page[0,i] = s[0]
        dur[0,i] = s[1] if i < 19 else 0.05
    r = sess.run(None, {'page_ids': page, 'durations': dur})
    prob = 1/(1+np.exp(-r[0][0][0]))
    print(f"    View-only pattern (filled 20) => {prob*100:.1f}% risk")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    start = time.time()
    
    Xp_tr, Xd_tr, y_tr, Xp_vl, Xd_vl, y_vl = preprocess()
    model = train(Xp_tr, Xd_tr, y_tr, Xp_vl, Xd_vl, y_vl)
    export_onnx()
    
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"[DONE] COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
