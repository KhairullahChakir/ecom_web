"""
Preprocess RetailRocket dataset into sequences for abandonment prediction.
Similar to the paper's format: sequences of (page_type, duration) with abandon labels.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import pickle

print("Loading RetailRocket events.csv...")
df = pd.read_csv('data/events.csv')
print(f"Total events: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nEvent types:\n{df['event'].value_counts()}")

# Sort by visitor and timestamp
df = df.sort_values(['visitorid', 'timestamp'])

# Map events to page types (like the paper)
EVENT_TO_PAGE = {
    'view': 1,           # Product view
    'addtocart': 2,      # Cart operation
    'transaction': 3     # Purchase (completed)
}

df['page_type'] = df['event'].map(EVENT_TO_PAGE)

# Group by visitor to create sessions
# A session ends when there's a gap > 30 minutes or a transaction
print("\nCreating sessions...")

sessions = []
session_labels = []  # 0 = purchased, 1 = abandoned

current_visitor = None
current_session = []
last_timestamp = None
SESSION_GAP_MS = 30 * 60 * 1000  # 30 minutes in milliseconds

for idx, row in df.iterrows():
    visitor = row['visitorid']
    timestamp = row['timestamp']
    page_type = row['page_type']
    event = row['event']
    
    # New visitor or session gap
    if visitor != current_visitor or (last_timestamp and timestamp - last_timestamp > SESSION_GAP_MS):
        # Save previous session
        if len(current_session) >= 2:  # At least 2 events
            sessions.append(current_session)
            # Label: abandoned if no transaction in session
            has_transaction = any(e[0] == 3 for e in current_session)
            session_labels.append(0 if has_transaction else 1)
        
        current_session = []
        current_visitor = visitor
    
    # Calculate duration (time until next event, or 0 for last)
    duration = 0  # Will be filled later
    current_session.append((page_type, duration, timestamp))
    last_timestamp = timestamp
    
    # Print progress
    if idx % 500000 == 0:
        print(f"  Processed {idx:,} events, {len(sessions):,} sessions...")

# Save last session
if len(current_session) >= 2:
    sessions.append(current_session)
    has_transaction = any(e[0] == 3 for e in current_session)
    session_labels.append(0 if has_transaction else 1)

print(f"\nTotal sessions: {len(sessions):,}")
print(f"Abandoned sessions: {sum(session_labels):,} ({100*sum(session_labels)/len(session_labels):.1f}%)")
print(f"Purchased sessions: {len(session_labels) - sum(session_labels):,}")

# Calculate durations within each session
print("\nCalculating durations...")
for session in sessions:
    for i in range(len(session) - 1):
        page_type, _, ts = session[i]
        next_ts = session[i + 1][2]
        duration = (next_ts - ts) / 1000  # Convert to seconds
        duration = min(duration, 600)  # Cap at 10 minutes
        session[i] = (page_type, duration)
    
    # Last event gets average duration
    session[-1] = (session[-1][0], 30.0)

# Prepare for training - pad sequences
MAX_SEQ_LEN = 20
print(f"\nPadding sequences to length {MAX_SEQ_LEN}...")

X_page = np.zeros((len(sessions), MAX_SEQ_LEN), dtype=np.int64)
X_dur = np.zeros((len(sessions), MAX_SEQ_LEN), dtype=np.float32)
y = np.array(session_labels, dtype=np.float32)

for i, session in enumerate(sessions):
    seq_len = min(len(session), MAX_SEQ_LEN)
    for j in range(seq_len):
        X_page[i, j] = session[j][0]
        X_dur[i, j] = session[j][1] / 600.0  # Normalize to [0, 1]

# Save
print("\nSaving processed data...")
np.save('data/X_page_real.npy', X_page)
np.save('data/X_dur_real.npy', X_dur)
np.save('data/y_abandon_real.npy', y)

print(f"\n✅ Saved:")
print(f"   X_page_real.npy: {X_page.shape}")
print(f"   X_dur_real.npy: {X_dur.shape}")
print(f"   y_abandon_real.npy: {y.shape}")
print(f"\n📊 Class distribution:")
print(f"   Abandoned: {y.sum():.0f} ({100*y.mean():.1f}%)")
print(f"   Purchased: {len(y) - y.sum():.0f} ({100*(1-y.mean()):.1f}%)")
