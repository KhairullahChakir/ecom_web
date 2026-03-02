import numpy as np
import os

DATA_DIR = "d:/op_ecom/data/processed"
X_page = np.load(os.path.join(DATA_DIR, "X_page_real.npy"))
y = np.load(os.path.join(DATA_DIR, "y_abandon_real.npy"))

print(f"Total samples: {len(y)}")
print(f"Abandonment rate: {y.mean()*100:.2f}%")

# Check if index 3 (Transaction) exists in X_page
has_3 = (X_page == 3).any(axis=1)
print(f"Samples containing Transaction (index 3) in input: {has_3.sum()}")

# Check correlation of AddToCart (index 2) with label
has_2 = (X_page == 2).any(axis=1)
print(f"Samples containing AddToCart (index 2) in input: {has_2.sum()}")

# Confusion matrix for has_2 vs y (abandon=1, purchase=0)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, has_2)
print("\nConfusion Matrix (Rows: y_true(0=Purch, 1=Aband), Cols: has_AddToCart):")
print(cm)

# If has_2 perfectly predicts 0, then cm[0,0] should be 0 and cm[1,1] should be 1, etc.
# Actually if has_2 -> purchase (y=0), then:
# y=0, has_2=True (col 1)
# y=1, has_2=False (col 0)
