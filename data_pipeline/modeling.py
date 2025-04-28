# Modeling

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data_pipeline.feature_engineering import load_data, create_team_features, prepare_features
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from utils.probability_adjustments import adjust_match_probabilities
from config.model_config import PREDICTION_FACTORS
from config.team_mapping import get_historical_name
import sqlite3
from config.model_config import (
    RECENT_MATCHES_WINDOW,
    WEIGHTS,
    DEFAULT_PROBABILITIES
)

TEAM_NAME_MAPPING = {
    # Current to Historical mappings
    'AFC Bournemouth': 'Bournemouth',
    'Brighton & Hove Albion': 'Brighton',
    'Leeds United': 'Leeds',
    'Nottingham Forest': 'Nottingham Forest',
    'Manchester United': 'Manchester United',
    'Manchester City': 'Manchester City',
    'Newcastle United': 'Newcastle United',
    'Tottenham Hotspur': 'Tottenham Hotspur',
    'West Ham United': 'West Ham United',
    
    # Keep these as direct mappings
    'Arsenal': 'Arsenal',
    'Aston Villa': 'Aston Villa',
    'Chelsea': 'Chelsea',
    'Liverpool': 'Liverpool',
    'Everton': 'Everton'
}

def get_historical_name(current_name: str) -> str:
    """Convert current team name to historical database name"""
    return TEAM_NAME_MAPPING.get(current_name, current_name)

def get_current_name(historical_name: str) -> str:
    """Convert historical database name to current team name"""
    reverse_mapping = {v: k for k, v in TEAM_NAME_MAPPING.items()}
    return reverse_mapping.get(historical_name, historical_name)

class SoccerPredictor:
    def __init__(self):
        """Initialize the soccer predictor"""
        self.model = None
        self.teams_data = None
        self.label_encoder = None
        self.feature_columns = None
        self.db = None
        self.db_path = None

    def train_model(self, db_path):
        """Train an enhanced prediction model"""
        # Store db_path for later connections
        self.db_path = db_path
        
        # Create database connection
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        
        # Load and prepare data
        matches, team_attrs, teams = load_data(db_path)
        processed_matches = create_team_features(matches, team_attrs, teams)
        X, y = prepare_features(processed_matches)
        
        # Define the feature columns in a specific order
        self.feature_columns = [
            'home_goals_rolling_mean',
            'away_goals_rolling_mean', 
            'home_team_form',
            'away_team_form',
            'buildUpPlaySpeed',
            'buildUpPlaySpeed_away',
            'buildUpPlayPassing',
            'buildUpPlayPassing_away',
            'chanceCreationPassing',
            'chanceCreationPassing_away',
            'chanceCreationShooting',
            'chanceCreationShooting_away',
            'defencePressure',
            'defencePressure_away',
            'head_to_head_wins',
            'head_to_head_draws',
            'home_goals_per_game',
            'home_conceded_per_game',
            'home_win_ratio'
        ]
        
        # Ensure X has features in the correct order
        X = X[self.feature_columns]
        
        # Encode target labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Store label encoder for predictions
        self.label_encoder = label_encoder
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        # Define XGBoost classifier with best parameters
        xgb_clf = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        # Train the model
        xgb_clf.fit(X_train_scaled, y_train)
        
        # Save the preprocessing steps and model
        self.model = {
            'imputer': imputer,
            'scaler': scaler,
            'classifier': xgb_clf
        }
        
        # Evaluate model
        y_pred = xgb_clf.predict(X_test_scaled)
        print("\nModel Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=label_encoder.classes_))
        
        # Store teams data for predictions
        self.teams_data = {
            'matches': processed_matches,
            'teams': teams,
            'team_attrs': team_attrs
        }
    
    def get_kaggle_team_name(self, rapidapi_team_name):
        """Convert current team name to Kaggle database team name"""
        # Remove common prefixes/suffixes
        name_variations = {
            'AFC ': '',  # For AFC Bournemouth -> Bournemouth
            'FC ': '',   # For FC teams
            '& Hove': '' # For Brighton & Hove Albion -> Brighton
        }
        
        db_name = rapidapi_team_name
        for prefix, replacement in name_variations.items():
            if db_name.startswith(prefix):
                db_name = db_name.replace(prefix, replacement)
        
        # If still no match, try the original mapping
        return TEAM_NAME_MAPPING.get(rapidapi_team_name, db_name)

    def _get_raw_predictions(self, home_team: str, away_team: str) -> tuple:
        """
        Get raw prediction probabilities for a match
        
        Args:
            home_team (str): Name of home team
            away_team (str): Name of away team
            
        Returns:
            tuple: (home_win_prob, draw_prob, away_win_prob)
        """
        try:
            # Ensure database connection exists
            if self.db is None:
                if self.db_path is None:
                    raise ValueError("Model not trained. Call train_model first.")
                self.db = sqlite3.connect(self.db_path)
                self.db.row_factory = sqlite3.Row
                
            # Convert team names to historical names
            home_team = get_historical_name(home_team)
            away_team = get_historical_name(away_team)
            
            # Get historical match data
            matches_query = """
                SELECT 
                    COUNT(*) as total_matches,
                    SUM(CASE WHEN home_team_goal > away_team_goal THEN 1 ELSE 0 END) as home_wins,
                    SUM(CASE WHEN home_team_goal = away_team_goal THEN 1 ELSE 0 END) as draws,
                    SUM(CASE WHEN home_team_goal < away_team_goal THEN 1 ELSE 0 END) as away_wins
                FROM Match
                WHERE home_team_api_id IN (
                    SELECT team_api_id 
                    FROM Team 
                    WHERE team_long_name = ?
                )
                AND away_team_api_id IN (
                    SELECT team_api_id 
                    FROM Team 
                    WHERE team_long_name = ?
                )
            """
            
            result = self.db.execute(matches_query, (home_team, away_team)).fetchone()
            
            if result and result['total_matches'] > 0:
                total = float(result['total_matches'])
                home_prob = result['home_wins'] / total
                draw_prob = result['draws'] / total
                away_prob = result['away_wins'] / total
            else:
                # If no historical matches, use league averages
                home_prob = 0.45  # Typical home win probability
                draw_prob = 0.25  # Typical draw probability
                away_prob = 0.30  # Typical away win probability
            
            return home_prob, draw_prob, away_prob
            
        except Exception as e:
            print(f"Error getting raw predictions: {e}")
            # Return reasonable default probabilities
            return 0.45, 0.25, 0.30

    def _get_historical_stats(self, home_team: str, away_team: str) -> dict:
        """Get historical head-to-head stats"""
        query = """
            SELECT 
                AVG(CASE WHEN home_team_goal > away_team_goal THEN 1 
                         WHEN home_team_goal = away_team_goal THEN 0.5
                         ELSE 0 END) as home_win_rate,
                COUNT(*) as total_matches
            FROM Match 
            WHERE (home_team_api_id IN (SELECT team_api_id FROM Team WHERE team_long_name = ?)
            AND away_team_api_id IN (SELECT team_api_id FROM Team WHERE team_long_name = ?))
            OR (home_team_api_id IN (SELECT team_api_id FROM Team WHERE team_long_name = ?)
            AND away_team_api_id IN (SELECT team_api_id FROM Team WHERE team_long_name = ?))
        """
        result = self.db.execute(query, (home_team, away_team, away_team, home_team)).fetchone()
        return {
            'win_rate': result['home_win_rate'] if result['home_win_rate'] is not None else 0.5,
            'matches': result['total_matches']
        }

    def _get_team_form(self, team_name: str) -> float:
        """Get team's current form based on recent matches"""
        try:
            # Convert team name to historical format
            team_name = get_historical_name(team_name)
            
            # Query recent matches
            query = """
                SELECT 
                    CASE 
                        WHEN (home_team_api_id IN (SELECT team_api_id FROM Team WHERE team_long_name = ?)
                             AND home_team_goal > away_team_goal)
                        OR (away_team_api_id IN (SELECT team_api_id FROM Team WHERE team_long_name = ?)
                             AND away_team_goal > home_team_goal) THEN 3
                        WHEN home_team_goal = away_team_goal THEN 1
                        ELSE 0 
                    END as points,
                    date
                FROM Match
                WHERE home_team_api_id IN (SELECT team_api_id FROM Team WHERE team_long_name = ?)
                OR away_team_api_id IN (SELECT team_api_id FROM Team WHERE team_long_name = ?)
                ORDER BY date DESC
                LIMIT ?
            """
            
            results = self.db.execute(query, (team_name, team_name, team_name, team_name, 
                                            RECENT_MATCHES_WINDOW)).fetchall()
            
            if not results:
                return 0.0
                
            # Calculate weighted form (more recent matches have higher weight)
            total_weight = 0
            weighted_points = 0
            for i, match in enumerate(results):
                weight = 1.0 / (i + 1)  # Decreasing weights for older matches
                weighted_points += match['points'] * weight
                total_weight += weight
                
            return weighted_points / (total_weight * 3)  # Normalize to 0-1 range
            
        except Exception as e:
            print(f"Error calculating team form: {e}")
            return 0.0

    def predict_match(self, home_team: str, away_team: str) -> dict:
        """Predict match outcome probabilities with weighted factors"""
        try:
            # Get raw historical predictions
            home_prob, draw_prob, away_prob = self._get_raw_predictions(home_team, away_team)
            
            # Get current form
            home_form = self._get_team_form(home_team)
            away_form = self._get_team_form(away_team)
            
            # Calculate weighted probabilities
            weighted_home = (
                WEIGHTS['historical'] * home_prob +
                WEIGHTS['current_form'] * home_form +
                WEIGHTS['home_advantage'] * DEFAULT_PROBABILITIES['home_win'] +
                WEIGHTS['recent_matches'] * home_form
            )
            
            weighted_away = (
                WEIGHTS['historical'] * away_prob +
                WEIGHTS['current_form'] * away_form +
                WEIGHTS['recent_matches'] * away_form
            )
            
            weighted_draw = 1 - (weighted_home + weighted_away)
            
            # Normalize probabilities
            total = weighted_home + weighted_draw + weighted_away
            return {
                'home_win': weighted_home / total,
                'draw': weighted_draw / total, 
                'away_win': weighted_away / total
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return DEFAULT_PROBABILITIES

    def get_available_teams(self):
        """Get list of available teams from the Kaggle database"""
        if self.teams_data is None:
            raise ValueError("Model not trained. Call train_model first.")
        return sorted(self.teams_data['teams']['team_long_name'].unique())

    def predict_league_standings(self, league_name, num_games=38):
        """Predict league standings using only Kaggle database teams"""
        if self.teams_data is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Get teams from specific league in Kaggle database
        teams_df = self.teams_data['teams']
        
        # Print available leagues for debugging
        print("\nAvailable leagues in database:")
        for col in teams_df.columns:
            print(col)
        
        # Most likely the column is named differently, try common variations
        league_column = None
        for col in ['league', 'league_id', 'name', 'league_name']:
            if col in teams_df.columns:
                league_column = col
                break
        
        if not league_column:
            raise ValueError("League column not found in database")
        
        league_teams = teams_df[teams_df[league_column] == league_name]['team_long_name'].tolist()
        
        if not league_teams:
            print(f"\nNo teams found for league: {league_name}")
            print("Available leagues:")
            print(teams_df[league_column].unique())
            raise ValueError(f"No teams found for league: {league_name}")
        
        # ... rest of the existing standings prediction code ...

    def __del__(self):
        """Cleanup database connection"""
        if self.db:
            self.db.close()

# Usage example:
if __name__ == "__main__":
    predictor = SoccerPredictor()
    predictor.train_model("data/raw/kaggle/soccer/database.sqlite")
    
    # Example predictions
    print(predictor.predict_match("Manchester United", "Chelsea"))
    
    premier_league_teams = [
        "Manchester United", "Chelsea", "Arsenal", "Liverpool",
        "Manchester City", "Tottenham", "Leicester", "West Ham"
    ]
    print(predictor.predict_league_standings("Premier League"))
