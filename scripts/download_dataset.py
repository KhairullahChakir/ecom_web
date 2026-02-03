"""Try different datasets"""
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Try alternative clickstream datasets
datasets_to_try = [
    'retailrocket/ecommerce-dataset',
    'mkechinov/ecommerce-behavior-data-from-multi-category-store',
    'arashsafavi/ecommerce-data',
]

for dataset in datasets_to_try:
    try:
        print(f"\nTrying: {dataset}")
        api.dataset_download_files(dataset, path='data', unzip=True)
        print(f"SUCCESS: Downloaded {dataset}")
        break
    except Exception as e:
        print(f"Failed: {type(e).__name__} - {str(e)[:50]}")
