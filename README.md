#Big Data Analytics in Soccer: Performance Insights and Predictive Modeling


A machine learning system for analyzing player performance and predicting soccer match outcomes, simulating league standings using historical Premier League data.

## Prerequisites

- Python 3.8+
- SQLite3
- Required Python packages:
  - pandas
  - scikit-learn
  - xgboost

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/soccer-prediction.git
cd soccer-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the required datasets:
run python ingestion.py

## Project Structure

```
finaldata/
├── data/
│   ├── raw/
│   │   ├── kaggle/
│   │   │   └── soccer/
│   │   │       └── database.sqlite
│   │   └── rapidapi/
│   │       └── premier_league/
│   │           └── premier_league.csv
│   └── processed/
├── data_pipeline/
│   ├── __init__.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── cleaning.py
├── predict_soccer.py
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
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\activate
# On Windows (Command Prompt):
venv\Scripts\activate.bat
# On Unix/MacOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install base requirements
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


### 5. Run the Data Pipeline

#### a. Data Cleaning
First, run the data cleaning notebook to process Kaggle data:
```bash
# Start Jupyter Notebook
jupyter notebook

# Open and run dataCleanKaggle.ipynb in the browser
```

This will:
1. Load and combine player statistics, valuations, and transfers
2. Clean and process the data
3. Save results to `data/processed/combined_kaggle.csv`

#### b. Feature Engineering
Run the feature engineering pipeline:
```bash
python data_pipeline/feature_engineering.py
```

This will:
1. Create team performance features (form, goals, win rates)
2. Generate head-to-head statistics
3. Calculate league positions and points
4. Add match importance indicators
5. Save engineered features to `data/processed/match_features.csv`

#### c. Main Pipeline
Then run the main pipeline:
```bash
python main.py
```

2. Or run the prediction system directly:
```bash
python predict_soccer.py
```

### Available Options

The system offers three main options:

1. **Predict Single Match**
   - Select two teams from the available list
   - Get win/draw/loss probabilities

2. **Predict League Standings**
   - Choose number of games to simulate (1-38)
   - Get predicted standings table with:
     - Points
     - Wins/Draws/Losses
     - Goal difference
     - Final position

3. **Exit**

## Team Name Mappings

The system automatically maps current team names to their historical counterparts in the Kaggle database. Current mappings include:

```python
TEAM_NAME_MAPPINGS = {
    'Arsenal': 'Arsenal',
    'Aston Villa': 'Aston Villa',
    'AFC Bournemouth': 'Bournemouth',
    'Brighton & Hove Albion': 'Brighton & Hove Albion',
    'Chelsea': 'Chelsea',
}
```

Add more mappings as needed in `predict_soccer.py`.

## Data Pipeline

1. **Data Ingestion**
   - Loads historical match data from Kaggle SQLite database
   - Loads current season data from RapidAPI

2. **Feature Engineering**
   - Creates team performance features
   - Calculates rolling averages and form indicators

3. **Model Training**
   - Uses XGBoost classifier
   - Trained on historical match outcomes

4. **Prediction**
   - Single match prediction
   - League standings simulation

## Error Handling

- The system validates team names against the database
- Provides debugging information for team name mapping
- Handles missing data and invalid inputs

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
