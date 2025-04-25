import pandas as pd
from typing import Dict, Set

def fetch_current_premier_league_teams() -> Set[str]:
    """
    Fetch current Premier League teams from CSV file
    Returns set of current team names
    """
    try:
        csv_path = "data/raw/rapidapi/premier_league/premier_league.csv"
        df = pd.read_csv(csv_path)
        return set(df['team_name'].unique())
    except Exception as e:
        print(f"Error reading Premier League CSV: {e}")
        return set()