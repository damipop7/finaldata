# BLUEDAM Soccer Data Analysis Project

## Overview

This project analyzes player performance and predicts match outcomes for the top five European soccer leagues by combining **historical data** from Kaggle and **live data** from RapidAPI.

## Project Structure
```
finaldata/
├── clients/
│   ├── __init__.py
│   ├── kaggle_client.py
│   └── rapidapi_client.py
├── configs/
│   └── config.yaml
├── data/
│   └── raw/
│       ├── kaggle/
│       │   ├── soccer/
│       │   ├── big-five-european-soccer-leagues/
│       │   └── player-scores/
│       └── rapidapi/
│           ├── premier_league/
│           ├── la_liga/
│           ├── bundesliga/
│           ├── serie_a/
│           ├── ligue_1/
│           └── ingestion_log.json
├── data_pipeline/
│   ├── __init__.py
│   └── ingestion.py
├── utils/
│   ├── __init__.py
│   └── log_utils.py
├── tests/
│   └── test_ingestion.py
├── main.py
└── requirements.txt
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/BLUEDAM.git
cd BLUEDAM/finaldata
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Credentials

Create `configs/config.yaml` with your API credentials:
```yaml
# Kaggle API credentials
KAGGLE_USERNAME: "your_kaggle_username"
KAGGLE_KEY: "your_kaggle_api_key"

# RapidAPI credentials
RAPIDAPI_KEY: "your_rapidapi_key"
```

To get API keys:
- **Kaggle**: Go to kaggle.com > Account > Create API Token
- **RapidAPI**: 
  1. Sign up at [RapidAPI](https://rapidapi.com)
  2. Subscribe to [API-Football API](https://rapidapi.com/api-sports/api/api-football/)
  3. Copy your API key

### 5. Run the Pipeline
```bash
python main.py
```

This will:
1. Check last data update time
2. Download Kaggle datasets (if not present)
3. Fetch current league standings
4. Save data in appropriate directories
5. Update the ingestion log

## Data Sources

### Historical Data (Kaggle)
- European Soccer Database
- Big Five European Soccer Leagues
- Player Scores Dataset

### Live Data (RapidAPI)
Current season standings for:
- Premier League
- La Liga
- Bundesliga
- Serie A
- Ligue 1

## Checking Data Updates

To check when data was last updated:
```python
from utils.log_utils import check_last_update
print(check_last_update())
```

## Running Tests
```bash
python -m pytest tests/
```

## Security Note
- Do not commit `config.yaml` with real credentials
- Consider using environment variables for sensitive data
- Add `config.yaml` to `.gitignore`
