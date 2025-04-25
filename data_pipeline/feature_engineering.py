import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(db_path):
    """Load data from SQLite database"""
    conn = sqlite3.connect(db_path)
    
    # Load relevant tables
    matches = pd.read_sql("SELECT * FROM Match", conn)
    team_attrs = pd.read_sql("SELECT * FROM Team_Attributes", conn)
    teams = pd.read_sql("SELECT * FROM Team", conn)
    
    conn.close()
    return matches, team_attrs, teams

def calculate_team_form(match_results, team_id, window=5):
    """Calculate team form based on recent results"""
    # Convert results to numerical values
    results = pd.Series(match_results)
    numeric_results = results.map({'win': 1, 'draw': 0.5, 'loss': 0})
    
    # Calculate rolling mean of results
    return numeric_results.rolling(window=window, min_periods=1).mean()

def create_team_features(matches, team_attrs, teams, window=5):
    """Create enhanced team performance features"""
    
    # Merge team names
    matches = matches.merge(teams, left_on='home_team_api_id', right_on='team_api_id', how='left')
    matches = matches.merge(teams, left_on='away_team_api_id', right_on='team_api_id', how='left', 
                          suffixes=('_home', '_away'))
    
    # Sort matches by date
    matches['date'] = pd.to_datetime(matches['date'])
    matches = matches.sort_values('date')
    
    # Create target variable
    matches['target'] = matches.apply(lambda x: 
                                    'win' if x['home_team_goal'] > x['away_team_goal']
                                    else 'draw' if x['home_team_goal'] == x['away_team_goal']
                                    else 'loss', axis=1)
    
    # Calculate rolling goals
    for team_id in matches['home_team_api_id'].unique():
        # Home matches
        home_matches = matches[matches['home_team_api_id'] == team_id]
        matches.loc[matches['home_team_api_id'] == team_id, 'home_goals_rolling_mean'] = \
            home_matches['home_team_goal'].rolling(window=window, min_periods=1).mean()
        
        # Away matches
        away_matches = matches[matches['away_team_api_id'] == team_id]
        matches.loc[matches['away_team_api_id'] == team_id, 'away_goals_rolling_mean'] = \
            away_matches['away_team_goal'].rolling(window=window, min_periods=1).mean()
        
        # Calculate form for home games
        home_results = matches[matches['home_team_api_id'] == team_id]['target']
        matches.loc[matches['home_team_api_id'] == team_id, 'home_team_form'] = \
            calculate_team_form(home_results, team_id, window)
        
        # Calculate form for away games
        away_results = matches[matches['away_team_api_id'] == team_id]['target'].map({'win': 'loss', 'loss': 'win', 'draw': 'draw'})
        matches.loc[matches['away_team_api_id'] == team_id, 'away_team_form'] = \
            calculate_team_form(away_results, team_id, window)
    
    # Add team attributes
    latest_attrs = team_attrs.sort_values('date').groupby('team_api_id').last()
    
    # Merge home team attributes
    matches = matches.merge(
        latest_attrs[['buildUpPlaySpeed', 'buildUpPlayPassing', 'chanceCreationPassing',
                     'chanceCreationShooting', 'defencePressure']],
        left_on='home_team_api_id',
        right_index=True,
        how='left'
    )
    
    # Merge away team attributes
    matches = matches.merge(
        latest_attrs[['buildUpPlaySpeed', 'buildUpPlayPassing', 'chanceCreationPassing',
                     'chanceCreationShooting', 'defencePressure']],
        left_on='away_team_api_id',
        right_index=True,
        how='left',
        suffixes=('', '_away')
    )
    
    # Add more sophisticated features
    for team_id in matches['home_team_api_id'].unique():
        # Last N matches performance
        team_matches = pd.concat([
            matches[matches['home_team_api_id'] == team_id],
            matches[matches['away_team_api_id'] == team_id]
        ]).sort_values('date')
        
        # Goals stats
        matches.loc[matches['home_team_api_id'] == team_id, 'home_goals_per_game'] = \
            team_matches['home_team_goal'].rolling(window=window, min_periods=1).mean()
        matches.loc[matches['home_team_api_id'] == team_id, 'home_conceded_per_game'] = \
            team_matches['away_team_goal'].rolling(window=window, min_periods=1).mean()
        
        # Form calculation (wins=3, draws=1, losses=0)
        team_matches['points'] = team_matches.apply(
            lambda x: 3 if x['home_team_goal'] > x['away_team_goal']
            else 1 if x['home_team_goal'] == x['away_team_goal']
            else 0, axis=1
        )
        matches.loc[matches['home_team_api_id'] == team_id, 'home_win_ratio'] = \
            team_matches['points'].rolling(window=window, min_periods=1).mean() / 3
    
    # Add head-to-head features
    matches['head_to_head_wins'] = 0
    matches['head_to_head_draws'] = 0
    
    for idx, match in matches.iterrows():
        h2h_matches = matches[
            ((matches['home_team_api_id'] == match['home_team_api_id']) & 
             (matches['away_team_api_id'] == match['away_team_api_id'])) |
            ((matches['home_team_api_id'] == match['away_team_api_id']) & 
             (matches['away_team_api_id'] == match['home_team_api_id']))
        ]
        
        matches.loc[idx, 'head_to_head_wins'] = len(h2h_matches[
            h2h_matches['home_team_goal'] > h2h_matches['away_team_goal']
        ])
        matches.loc[idx, 'head_to_head_draws'] = len(h2h_matches[
            h2h_matches['home_team_goal'] == h2h_matches['away_team_goal']
        ])
    
    return matches

def prepare_features(matches):
    """Prepare final feature set"""
    feature_columns = [
        'home_goals_rolling_mean', 'away_goals_rolling_mean',
        'home_team_form', 'away_team_form',
        'buildUpPlaySpeed', 'buildUpPlaySpeed_away',
        'buildUpPlayPassing', 'buildUpPlayPassing_away',
        'chanceCreationPassing', 'chanceCreationPassing_away',
        'chanceCreationShooting', 'chanceCreationShooting_away',
        'defencePressure', 'defencePressure_away',
        'home_goals_per_game', 'home_conceded_per_game',
        'home_win_ratio', 'head_to_head_wins', 'head_to_head_draws'
    ]
    
    X = matches[feature_columns].copy()
    y = matches['target']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y

def main():
    # Test the feature engineering pipeline
    matches, team_attrs, teams = load_data("data/raw/kaggle/soccer/database.sqlite")
    match_features = create_team_features(matches, team_attrs, teams)
    X, y = prepare_features(match_features)
    print("Feature engineering completed successfully!")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

if __name__ == "__main__":
    main()