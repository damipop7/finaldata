import requests
import yaml
import os
import json
import time
import pandas as pd
from datetime import datetime

def fetch_rapidapi_league_data(base_path="data/raw/rapidapi"):
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if isinstance(config, dict) and "RAPIDAPI_KEY" in config:
        rapidapi_key = config["RAPIDAPI_KEY"]
    else:
        raise ValueError("Invalid or missing RAPIDAPI_KEY in the config.yaml file.")

    url = "https://api-football-v1.p.rapidapi.com/v3/standings"
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }

    leagues = {
        "premier_league": {"id": 39, "name": "Premier League"},
        "la_liga": {"id": 140, "name": "La Liga"},
        "bundesliga": {"id": 78, "name": "Bundesliga"},
        "serie_a": {"id": 135, "name": "Serie A"},
        "ligue_1": {"id": 61, "name": "Ligue 1"}
    }

    for league_key, league_info in leagues.items():
        print(f"🌍 Fetching standings for {league_info['name']}...")
        league_data = []
        
        querystring = {"season": "2024", "league": str(league_info['id'])}
        response = requests.get(url, headers=headers, params=querystring)

        if response.status_code == 200:
            data = response.json()
            standings = data['response'][0]['league']['standings'][0]
            
            for team in standings:
                team_data = {
                    'league': league_info['name'],
                    'rank': team['rank'],
                    'team_name': team['team']['name'],
                    'points': team['points'],
                    'played': team['all']['played'],
                    'win': team['all']['win'],
                    'draw': team['all']['draw'],
                    'lose': team['all']['lose'],
                    'goals_diff': team['goalsDiff'],
                    'form': team['form'],
                    'last_update': team['update']
                }
                league_data.append(team_data)

            # Create league-specific directory
            league_dir = os.path.join(base_path, league_key)
            os.makedirs(league_dir, exist_ok=True)

            # Save league-specific CSV
            csv_path = os.path.join(league_dir, f'{league_key}.csv')
            df = pd.DataFrame(league_data)
            df.to_csv(csv_path, index=False)
            print(f"✅ {league_info['name']} data saved to {csv_path}")

        else:
            print(f"❌ Failed to fetch {league_info['name']}: {response.status_code}")

        time.sleep(1.2)

    # Update ingestion log
    update_ingestion_log(base_path)

def update_ingestion_log(base_path):
    """Update the ingestion log with timestamp"""
    log_path = os.path.join(base_path, 'ingestion_log.json')
    
    log_entry = {
        'last_update': datetime.now().isoformat(),
        'status': 'success',
        'leagues_updated': [
            'premier_league',
            'la_liga',
            'bundesliga',
            'serie_a',
            'ligue_1'
        ]
    }
    
    # Create or update log file
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_data = json.load(f)
            if not isinstance(log_data, list):
                log_data = []
    else:
        log_data = []
    
    log_data.append(log_entry)
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"✅ Ingestion log updated at {log_path}")
