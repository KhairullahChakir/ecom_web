import pandas as pd
import numpy as np
import os

# Paths
DATA_DIR = "d:/op_ecom/scripts/data"
INPUT_PATH = os.path.join(DATA_DIR, "events.csv")
OUTPUT_DIR = "d:/op_ecom/data/processed"

print(f"Loading raw data from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)

# Sort by visitor and timestamp for single-pass iteration
print("Sorting data...")
df = df.sort_values(['visitorid', 'timestamp'])

EVENT_TO_PAGE = {
    'view': 1,
    'addtocart': 2,
    'transaction': 3
}
df['page_type'] = df['event'].map(EVENT_TO_PAGE).fillna(0).astype(int)

# Use itertuples for much faster iteration than iterrows/groupby
def create_leakage_proof_sessions_fast(df, gap_minutes=30):
    gap_ms = gap_minutes * 60 * 1000
    sessions = []
    session_labels = []
    
    current_visitor = None
    current_session = []
    last_ts = None
    
    print("Iterating through events...")
    for row in df.itertuples():
        v_id = row.visitorid
        ts = row.timestamp
        pt = row.page_type
        
        # New visitor or session gap
        if v_id != current_visitor or (last_ts and ts - last_ts > gap_ms):
            # Process previous session
            if len(current_session) >= 1:
                has_tx = any(e[0] == 3 for e in current_session)
                filtered_seq = [e for e in current_session if e[0] != 3]
                if len(filtered_seq) >= 1:
                    sessions.append(filtered_seq)
                    session_labels.append(0 if has_tx else 1)
            
            # Reset
            current_session = []
            current_visitor = v_id
            
        current_session.append((pt, ts))
        last_ts = ts
        
    # Final session
    if len(current_session) >= 1:
        has_tx = any(e[0] == 3 for e in current_session)
        filtered_seq = [e for e in current_session if e[0] != 3]
        if len(filtered_seq) >= 1:
            sessions.append(filtered_seq)
            session_labels.append(0 if has_tx else 1)
            
    return sessions, session_labels

print("Generating sessions...")
sessions_raw, labels = create_leakage_proof_sessions_fast(df)
print(f"Generated {len(sessions_raw):,} sessions.")

# Padding and saving (standard length of 20)
MAX_LEN = 20
X_page = np.zeros((len(sessions_raw), MAX_LEN), dtype=np.int64)
X_dur = np.zeros((len(sessions_raw), MAX_LEN), dtype=np.float32)
y = np.array(labels, dtype=np.float32)

for i, session in enumerate(sessions_raw):
    for j in range(min(len(session), MAX_LEN)):
        X_page[i, j] = session[j][0]
        if j < len(session) - 1:
            dur = (session[j+1][1] - session[j][1]) / 1000
            X_dur[i, j] = min(dur, 600) / 600.0
        else:
            X_dur[i, j] = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "X_page_real.npy"), X_page)
np.save(os.path.join(OUTPUT_DIR, "X_dur_real.npy"), X_dur)
np.save(os.path.join(OUTPUT_DIR, "y_abandon_real.npy"), y)

print(f"✅ Saved leakage-proof data to {OUTPUT_DIR}/")
print(f"Abandonment Rate: {y.mean()*100:.2f}%")
