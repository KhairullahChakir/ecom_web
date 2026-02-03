"""
Sequential Clickstream Data Generator
Generates synthetic user sessions for training the Abandonment Prediction Transformer.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import random

# Page Types (matching our tracker)
PAGE_TYPES = ['Home', 'Product', 'ProductDetail', 'Cart', 'Checkout', 'About', 'Account']

# Behavior Profiles
PROFILES = {
    'buyer': {
        'description': 'High intent, focused on products, completes purchase',
        'page_weights': [0.1, 0.3, 0.25, 0.2, 0.1, 0.03, 0.02],
        'avg_duration': (15, 120),  # seconds per page
        'session_length': (5, 15),  # number of pages
        'abandonment_prob': 0.1,
    },
    'window_shopper': {
        'description': 'Browses many products but never buys',
        'page_weights': [0.15, 0.45, 0.3, 0.05, 0.0, 0.03, 0.02],
        'avg_duration': (5, 30),
        'session_length': (3, 10),
        'abandonment_prob': 0.9,
    },
    'bouncer': {
        'description': 'Quickly leaves after landing',
        'page_weights': [0.6, 0.2, 0.1, 0.0, 0.0, 0.05, 0.05],
        'avg_duration': (1, 10),
        'session_length': (1, 3),
        'abandonment_prob': 0.95,
    },
    'researcher': {
        'description': 'Spends time on About/Info pages, may or may not buy',
        'page_weights': [0.1, 0.2, 0.15, 0.1, 0.05, 0.3, 0.1],
        'avg_duration': (30, 180),
        'session_length': (4, 8),
        'abandonment_prob': 0.6,
    },
    'cart_abandoner': {
        'description': 'Gets to cart but leaves before checkout',
        'page_weights': [0.1, 0.25, 0.2, 0.35, 0.05, 0.03, 0.02],
        'avg_duration': (10, 60),
        'session_length': (4, 10),
        'abandonment_prob': 0.85,
    }
}


def generate_session(profile_name: str) -> Tuple[List[Tuple[str, float]], int]:
    """
    Generate a single user session based on a behavior profile.
    
    Returns:
        sequence: List of (page_type, duration_seconds) tuples
        label: 1 if abandoned, 0 if purchased
    """
    profile = PROFILES[profile_name]
    
    # Determine session length
    min_len, max_len = profile['session_length']
    session_length = random.randint(min_len, max_len)
    
    # Generate page sequence
    sequence = []
    for _ in range(session_length):
        # Pick page type based on weights
        page_type = np.random.choice(PAGE_TYPES, p=profile['page_weights'])
        
        # Generate duration
        min_dur, max_dur = profile['avg_duration']
        duration = random.uniform(min_dur, max_dur)
        
        # Add some noise to make it realistic
        duration = max(1.0, duration + random.gauss(0, 5))
        
        sequence.append((page_type, round(duration, 2)))
    
    # Determine outcome
    abandoned = random.random() < profile['abandonment_prob']
    
    return sequence, int(abandoned)


def generate_dataset(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a full dataset of user sessions.
    
    Returns:
        DataFrame with columns: session_id, sequence (as string), label
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Profile distribution (mimics real e-commerce traffic)
    profile_weights = {
        'buyer': 0.15,
        'window_shopper': 0.35,
        'bouncer': 0.25,
        'researcher': 0.10,
        'cart_abandoner': 0.15,
    }
    
    data = []
    for i in range(n_samples):
        # Pick a profile
        profile_name = np.random.choice(
            list(profile_weights.keys()),
            p=list(profile_weights.values())
        )
        
        # Generate session
        sequence, label = generate_session(profile_name)
        
        data.append({
            'session_id': i,
            'profile': profile_name,
            'sequence': sequence,
            'label': label
        })
    
    df = pd.DataFrame(data)
    
    # Print statistics
    print(f"Generated {len(df)} sessions")
    print(f"Abandonment Rate: {df['label'].mean()*100:.1f}%")
    print(f"\nProfile Distribution:")
    print(df['profile'].value_counts(normalize=True).round(2))
    
    return df


def prepare_for_training(df: pd.DataFrame, max_seq_len: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert DataFrame to padded numpy arrays for model training.
    
    Returns:
        X_page: (n_samples, max_seq_len) - Page type indices
        X_dur: (n_samples, max_seq_len) - Duration values (normalized)
        y: (n_samples,) - Labels
    """
    page_to_idx = {page: i+1 for i, page in enumerate(PAGE_TYPES)}  # 0 reserved for padding
    
    X_page = np.zeros((len(df), max_seq_len), dtype=np.int32)
    X_dur = np.zeros((len(df), max_seq_len), dtype=np.float32)
    y = df['label'].values
    
    for i, row in df.iterrows():
        seq = row['sequence']
        seq_len = min(len(seq), max_seq_len)
        
        for j, (page, dur) in enumerate(seq[:seq_len]):
            X_page[i, j] = page_to_idx[page]
            X_dur[i, j] = dur / 180.0  # Normalize to ~[0, 1]
    
    return X_page, X_dur, y


if __name__ == "__main__":
    # Generate dataset
    df = generate_dataset(n_samples=10000)
    
    # Save raw data
    df.to_pickle("data/sequential_clickstream.pkl")
    print(f"\nSaved to data/sequential_clickstream.pkl")
    
    # Prepare for training
    X_page, X_dur, y = prepare_for_training(df)
    
    # Save numpy arrays
    np.save("data/X_page.npy", X_page)
    np.save("data/X_dur.npy", X_dur)
    np.save("data/y_abandon.npy", y)
    
    print(f"\nTraining data shapes:")
    print(f"  X_page: {X_page.shape}")
    print(f"  X_dur: {X_dur.shape}")
    print(f"  y: {y.shape}")
    print(f"\nReady for Transformer training! 🚀")
