"""
Download UCI Online Shoppers Purchasing Intention Dataset
"""
import os
import urllib.request

# Dataset URL from UCI repository
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "online_shoppers_intention.csv")

def download_dataset():
    """Download the dataset if not exists"""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    if os.path.exists(OUTPUT_PATH):
        print(f"Dataset already exists at: {OUTPUT_PATH}")
        return OUTPUT_PATH
    
    print(f"Downloading dataset from UCI...")
    urllib.request.urlretrieve(DATASET_URL, OUTPUT_PATH)
    print(f"Downloaded to: {OUTPUT_PATH}")
    return OUTPUT_PATH

if __name__ == "__main__":
    download_dataset()
