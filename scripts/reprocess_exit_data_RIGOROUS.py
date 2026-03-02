import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = "d:/op_ecom/scripts/data"
INPUT_PATH = os.path.join(DATA_DIR, "events.csv")
OUTPUT_DIR = "d:/op_ecom/data/processed_rigorous"

print(f"Loading raw data from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)
df = df.sort_values(['visitorid', 'timestamp'])

# 1. Split by Visitor ID (Ensures no overlap)
all_visitors = df['visitorid'].unique()
v_train, v_test = train_test_split(all_visitors, test_size=0.2, random_state=42)
v_test, v_val = train_test_split(v_test, test_size=0.5, random_state=42)

set_train = set(v_train)
set_val = set(v_val)
set_test = set(v_test)

EVENT_TO_PAGE = {'view': 1, 'addtocart': 2, 'transaction': 3}
df['page_type'] = df['event'].map(EVENT_TO_PAGE).fillna(0).astype(int)

def process_rigorous_sessions(df, visitor_set, gap_minutes=30):
    gap_ms = gap_minutes * 60 * 1000
    X_page = []
    X_dur = []
    y = []
    
    current_visitor = None
    current_session = [] # List of (page_type, timestamp)
    last_ts = None
    
    # Filter DF for relevant visitors to speed up
    subset = df[df['visitorid'].isin(visitor_set)]
    
    for row in subset.itertuples():
        v_id = row.visitorid
        ts = row.timestamp
        pt = row.page_type
        
        if v_id != current_visitor or (last_ts and ts - last_ts > gap_ms):
            if len(current_session) >= 1:
                # Rigor check: Find first transaction
                events = [e[0] for e in current_session]
                if 3 in events:
                    # Purchase session: Truncate BEFORE first transaction
                    first_tx_idx = events.index(3)
                    final_seq = current_session[:first_tx_idx]
                    label = 0
                else:
                    # Abandon session: All events
                    final_seq = current_session
                    label = 1
                
                if len(final_seq) >= 1:
                    X_page.append([e[0] for e in final_seq])
                    # Calculate durations
                    durs = []
                    for k in range(len(final_seq)):
                        if k < len(final_seq) - 1:
                            d = (final_seq[k+1][1] - final_seq[k][1]) / 1000
                            durs.append(min(d, 600) / 600.0)
                        else:
                            durs.append(0.05)
                    X_dur.append(durs)
                    y.append(label)
                    
            current_session = []
            current_visitor = v_id
        
        current_session.append((pt, ts))
        last_ts = ts
        
    return X_page, X_dur, y

def pad_and_save(X_p_list, X_d_list, y_list, name):
    MAX_LEN = 20
    n = len(y_list)
    X_p = np.zeros((n, MAX_LEN), dtype=np.int64)
    X_d = np.zeros((n, MAX_LEN), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    for i in range(n):
        p_seq = X_p_list[i]
        d_seq = X_d_list[i]
        length = min(len(p_seq), MAX_LEN)
        X_p[i, :length] = p_seq[:length]
        X_d[i, :length] = d_seq[:length]
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, f"X_page_{name}.npy"), X_p)
    np.save(os.path.join(OUTPUT_DIR, f"X_dur_{name}.npy"), X_d)
    np.save(os.path.join(OUTPUT_DIR, f"y_{name}.npy"), y)
    print(f"Saved {name} set: {n} samples")

print("Processing Train Set...")
Xp_tr, Xd_tr, y_tr = process_rigorous_sessions(df, set_train)
pad_and_save(Xp_tr, Xd_tr, y_tr, "train")

print("Processing Val Set...")
Xp_vl, Xd_vl, y_vl = process_rigorous_sessions(df, set_val)
pad_and_save(Xp_vl, Xd_vl, y_vl, "val")

print("Processing Test Set...")
Xp_te, Xd_te, y_te = process_rigorous_sessions(df, set_test)
pad_and_save(Xp_te, Xd_te, y_te, "test")

print("\n--- Rigorous Data Stats ---")
print(f"Total Rigorous Sessions: {len(y_tr)+len(y_vl)+len(y_te):,}")
print(f"Total Purchase Sessions: {sum(y_tr==0)+sum(y_vl==0)+sum(y_te==0):,}")
print(f"Abandonment Rate: {(len(y_tr)+len(y_vl)+len(y_te) - (sum(y_tr==0)+sum(y_vl==0)+sum(y_te==0)))/(len(y_tr)+len(y_vl)+len(y_te))*100:.2f}%")
