import numpy as np
import os

DATA_DIR = "d:/op_ecom/data/processed"
X_page = np.load(os.path.join(DATA_DIR, "X_page_real.npy"))
y = np.load(os.path.join(DATA_DIR, "y_abandon_real.npy"))

# 0 is padding, 1=view, 2=addtocart, 3=transaction (but 3 is removed from X)
session_lengths = np.sum(X_page != 0, axis=1)

print(f"Dataset Analysis:")
print(f"Total Sessions: {len(X_page):,}")
print(f"Average Session Length: {np.mean(session_lengths):.2f} events")
print(f"Median Session Length: {np.median(session_lengths):.2f} events")
print(f"Max Sequence Length (Clipped): {X_page.shape[1]}")

# Event counts (excluding padding)
unique, counts = np.unique(X_page[X_page != 0], return_counts=True)
event_dist = dict(zip(unique, counts))

print("\nEvent Distribution (Input Features):")
print(f" - View (1): {event_dist.get(1, 0):,}")
print(f" - AddToCart (2): {event_dist.get(2, 0):,}")
print(f" - Transaction (3): {event_dist.get(3, 0):,} (Should be 0 due to leakage fix)")

# Target distribution
num_purchases = np.sum(y == 0)
num_abandons = np.sum(y == 1)
print(f"\nTarget Labels:")
print(f" - Purchases (0): {num_purchases:,} ({num_purchases/len(y)*100:.2f}%)")
print(f" - Abandons (1): {num_abandons:,} ({num_abandons/len(y)*100:.2f}%)")
