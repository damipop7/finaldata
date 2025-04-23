import os

def check_kaggle_data_exists(base_path="data/raw/kaggle"):
    """Check if all required Kaggle datasets are already downloaded"""
    required_folders = [
        "soccer",
        "big-five-european-soccer-leagues",
        "player-scores"
    ]
    
    for folder in required_folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path) or not os.listdir(folder_path):
            return False
    return True

def run_ingestion():
    """Run the data ingestion pipeline with checks for existing data"""
    print("🚀 Starting data ingestion...")
    
    # Check for existing Kaggle data
    if check_kaggle_data_exists():
        print("✅ Kaggle data already exists, skipping download")
    else:
        print("📥 Downloading Kaggle datasets...")
        from clients.kaggle_client import download_kaggle_datasets
        download_kaggle_datasets()

    # Always fetch fresh RapidAPI data as it's current league standings
    print("📊 Fetching current league standings...")
    from clients.rapidapi_client import fetch_rapidapi_league_data
    fetch_rapidapi_league_data()

