# Test_ingestion
import os
import pytest
import glob
from data_pipeline.ingestion import check_kaggle_data_exists

# Base data paths
KAGGLE_BASE = "data/raw/kaggle"
RAPIDAPI_PATH = "data/raw/rapidapi/league_tables.json"

# Expected Kaggle folders
KAGGLE_SUBFOLDERS = [
    "soccer",
    "big-five-european-soccer-leagues",
    "player-scores"
]

def test_kaggle_folders_exist():
    for folder in KAGGLE_SUBFOLDERS:
        path = os.path.join(KAGGLE_BASE, folder)
        assert os.path.exists(path), f"Kaggle folder missing: {path}"
        assert os.path.isdir(path), f"{path} is not a directory"

def test_kaggle_files_not_empty():
    for folder in KAGGLE_SUBFOLDERS:
        path = os.path.join(KAGGLE_BASE, folder)
        files = glob.glob(os.path.join(path, "*"))
        assert len(files) > 0, f"No files found in {path}"

def test_rapidapi_file_exists():
    assert os.path.exists(RAPIDAPI_PATH), "RapidAPI JSON file is missing"
    assert os.path.getsize(RAPIDAPI_PATH) > 100, "RapidAPI file is too small — might be empty or invalid"

def test_check_kaggle_data_exists():
    # Test when data exists
    assert check_kaggle_data_exists() == True, "Should return True when all required Kaggle data exists"

def test_check_kaggle_data_missing():
    # Test with non-existent path
    assert check_kaggle_data_exists("nonexistent/path") == False, "Should return False when path doesn't exist"

