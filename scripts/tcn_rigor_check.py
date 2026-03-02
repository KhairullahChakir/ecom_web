import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = "d:/op_ecom/scripts/data"
INPUT_PATH = os.path.join(DATA_DIR, "events.csv")

print("Loading data for rigor check...")
df = pd.read_csv(INPUT_PATH)
df = df.sort_values(['visitorid', 'timestamp'])

def get_session_info(df, gap_minutes=30):
    gap_ms = gap_minutes * 60 * 1000
    session_data = [] # List of (visitorid, has_purchase, length_before_first_tx, total_length_filtered)
    
    current_visitor = None
    current_session = []
    last_ts = None
    
    for row in df.itertuples():
        v_id = row.visitorid
        ts = row.timestamp
        pt = row.event
        
        if v_id != current_visitor or (last_ts and ts - last_ts > gap_ms):
            if len(current_session) >= 1:
                events = [e[0] for e in current_session]
                has_tx = 'transaction' in events
                
                # Length before first transaction
                if has_tx:
                    first_tx_idx = events.index('transaction')
                    len_before = first_tx_idx
                else:
                    len_before = len(events)
                
                filtered_len = len([e for e in events if e != 'transaction'])
                
                if filtered_len >= 1:
                    session_data.append({
                        'visitorid': current_visitor,
                        'has_purchase': has_tx,
                        'len_before': len_before,
                        'len_filtered': filtered_len,
                        'has_addtocart': 'addtocart' in events
                    })
            
            current_session = []
            current_visitor = v_id
        
        current_session.append((row.event, ts))
        last_ts = ts
        
    # Final session
    if len(current_session) >= 1:
        events = [e[0] for e in current_session]
        has_tx = 'transaction' in events
        if has_tx:
            first_tx_idx = events.index('transaction')
            len_before = first_tx_idx
        else:
            len_before = len(events)
        
        filtered_len = len([e for e in events if e != 'transaction'])
        if filtered_len >= 1:
            session_data.append({
                'visitorid': current_visitor,
                'has_purchase': has_tx,
                'len_before': len_before,
                'len_filtered': filtered_len,
                'has_addtocart': 'addtocart' in events
            })
            
    return pd.DataFrame(session_data)

print("Analyzing sessions...")
s_df = get_session_info(df)

# 1. Visitor Overlap Check
indices = np.arange(len(s_df))
train_idx, test_idx = train_test_split(indices, test_size=0.1, stratify=s_df['has_purchase'], random_state=42)

train_visitors = set(s_df.iloc[train_idx]['visitorid'])
test_visitors = set(s_df.iloc[test_idx]['visitorid'])
overlap = train_visitors.intersection(test_visitors)

print(f"\n--- Visitor Overlap ---")
print(f"Total Sessions: {len(s_df):,}")
print(f"Total Visitors: {s_df['visitorid'].nunique():,}")
print(f"Overlap Visitors: {len(overlap):,}")
print(f"Overlap % of Test Visitors: {len(overlap)/len(test_visitors)*100:.2f}%")

# 2. Behavioral Differences (Buyers vs Non-Buyers)
buyers = s_df[s_df['has_purchase']]
non_buyers = s_df[~s_df['has_purchase']]

print(f"\n--- Behavioral Profile ---")
print(f"Mean Length (Filtered) - Buyers: {buyers['len_filtered'].mean():.2f}")
print(f"Mean Length (Filtered) - Non-Buyers: {non_buyers['len_filtered'].mean():.2f}")
print(f"Median Length - Buyers: {buyers['len_filtered'].median()}")
print(f"Median Length - Non-Buyers: {non_buyers['len_filtered'].median()}")

print(f"\n--- Add-to-Cart Presence ---")
print(f"Buyers with AddToCart: {buyers['has_addtocart'].mean()*100:.2f}%")
print(f"Non-Buyers with AddToCart: {non_buyers['has_addtocart'].mean()*100:.2f}%")

# 3. Leakage Check: Length before first purchase
print(f"\n--- Purchase Point Analysis ---")
print(f"Mean Length before first Transaction: {buyers['len_before'].mean():.2f}")
print(f"Max items after Transaction in session: {(buyers['len_filtered'] - buyers['len_before']).max()}")
print(f"Sessions with items AFTER transaction: {(buyers['len_filtered'] > buyers['len_before']).sum()}")

