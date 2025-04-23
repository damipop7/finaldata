import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_datasets(base_path="data/raw/kaggle"):
    datasets = {
        "hugomathien/soccer": "soccer",
        "hikne707/big-five-european-soccer-leagues": "big-five-european-soccer-leagues",
        "davidcariboo/player-scores": "player-scores"
    }

    api = KaggleApi()
    api.authenticate()

    for dataset, folder_name in datasets.items():
        path = os.path.join(base_path, folder_name)
        os.makedirs(path, exist_ok=True)

        print(f"📥 Downloading {dataset} into {path}...")
        api.dataset_download_files(dataset, path=path, unzip=True)
        print(f"✅ {dataset} downloaded and extracted.")
