from data_pipeline.modeling import SoccerPredictor
import pandas as pd
from typing import Set, Dict, List

TEAM_NAME_MAPPINGS = {
    'Arsenal': 'Arsenal',
    'Aston Villa': 'Aston Villa',
    'AFC Bournemouth': 'Bournemouth',
    'Brighton & Hove Albion': 'Brighton & Hove Albion',
    'Chelsea': 'Chelsea',
}

TEAM_MAPPINGS = {
    'Nottingham Forest': 'Nottingham Forest',
    'Brentford': 'Brentford',
    'Brighton': 'Brighton & Hove Albion',
    'Brighton & Hove Albion': 'Brighton & Hove Albion',
    'Ipswich': 'Ipswich Town',
    'AFC Bournemouth': 'Bournemouth',
    'Wolverhampton': 'Wolverhampton Wanderers',
    'Wolves': 'Wolverhampton Wanderers',
    'Newcastle': 'Newcastle United',
    'Spurs': 'Tottenham Hotspur',
    'Tottenham': 'Tottenham Hotspur'
}

class PredictorSingleton:
    _instance = None
    
    @classmethod
    def get_predictor(cls):
        if cls._instance is None:
            print("\nInitializing prediction model (this may take a moment)...")
            cls._instance = SoccerPredictor()
            cls._instance.train_model("data/raw/kaggle/soccer/database.sqlite")
        return cls._instance

def get_available_teams(predictor):
    """Get list of available teams from the model"""
    return sorted(predictor.teams_data['teams']['team_long_name'].unique())

def print_available_teams(teams):
    """Print available teams in a formatted way"""
    print("\nAvailable teams:")
    for i, team in enumerate(teams, 1):
        print(f"{i}. {team}")

def get_team_selection(teams, prompt):
    """Get valid team selection from user"""
    while True:
        print_available_teams(teams)
        team = input(prompt)
        if team in teams:
            return team
        print(f"\nError: '{team}' not found. Please enter exact team name from the list.")

def verify_team_names(predictor, league_teams):
    """Verify all team names can be mapped to the Kaggle database"""
    print("\nVerifying team names...")
    kaggle_teams = set(predictor.teams_data['teams']['team_long_name'])
    
    for team in league_teams:
        kaggle_name = predictor.get_kaggle_team_name(team)
        if kaggle_name not in kaggle_teams:
            print(f"Warning: {team} -> {kaggle_name} not found in database")
            print("Available teams that match first few letters:")
            matches = [t for t in kaggle_teams if t.lower().startswith(kaggle_name.lower()[:4])]
            for match in matches:
                print(f"  - {match}")
    print()

def get_common_words(team_name: str) -> set:
    """
    Extract meaningful words from team name, converting to lowercase
    
    Args:
        team_name (str): Team name to process
        
    Returns:
        set: Set of processed words in lowercase
    """
    # Remove common suffixes and prefixes
    replacements = {
        'AFC': '',
        'FC': '',
        'United': '',
        'City': '',
        'Albion': '',
        '&': ' '  # Convert & to space for proper word splitting
    }
    
    # Convert to lowercase first for consistent processing
    name = team_name.lower()
    
    # Apply replacements
    for old, new in replacements.items():
        name = name.replace(old.lower(), new)
    
    # Split into words and remove empty strings
    return set(word.strip() for word in name.split() if word.strip())

def create_team_mapping(available_teams: Set[str], current_teams: Set[str]) -> Dict[str, str]:
    """Create mapping between current Premier League teams and historical database teams"""
    mapping = {}
    
    for current_team in current_teams:
        # Check predefined mappings first
        if current_team in TEAM_MAPPINGS:
            mapped_name = TEAM_MAPPINGS[current_team]
            mapping[current_team] = mapped_name
            continue
            
        # Direct match check
        if current_team in available_teams:
            mapping[current_team] = current_team
            continue
            
        # Word matching as fallback
        current_words = get_common_words(current_team)
        best_match = None
        max_common_words = 0
        
        for db_team in available_teams:
            db_words = get_common_words(db_team)
            common_words = len(current_words & db_words)
            
            if common_words > max_common_words:
                max_common_words = common_words
                best_match = db_team
                
        if best_match:
            mapping[current_team] = best_match
    
    return mapping

def predict_single_match():
    """Handle single match prediction workflow"""
    predictor = PredictorSingleton.get_predictor()
    
    print("\n=== Match Prediction ===")
    home_team = get_team_selection(predictor.get_available_teams(), "\nEnter home team: ")
    away_team = get_team_selection(predictor.get_available_teams(), "\nEnter away team: ")
    
    try:
        # Get raw predictions
        raw_result = predictor.predict_match(home_team, away_team)
        
        # Apply probability adjustments
        from utils.probability_adjustments import adjust_match_probabilities
        result = adjust_match_probabilities(
            raw_result['home_win'],
            raw_result['draw'],
            raw_result['away_win']
        )
        
        print(f"\nPrediction for {home_team} vs {away_team}:")
        print(f"Home win probability: {result['home_win']:.2%}")
        print(f"Draw probability: {result['draw']:.2%}")
        print(f"Away win probability: {result['away_win']:.2%}")
    except Exception as e:
        print(f"\nError predicting match: {e}")

def fetch_current_premier_league_teams():
    """Fetch current Premier League teams from CSV"""
    try:
        csv_path = "data/raw/rapidapi/premier_league/premier_league.csv"
        df = pd.read_csv(csv_path)
        return set(df['team_name'].unique())
    except Exception as e:
        print(f"Error reading Premier League CSV: {e}")
        return set()

def predict_league_standings():
    try:
        predictor = PredictorSingleton.get_predictor()
        
        # Load current Premier League data
        league_data = pd.read_csv("data/raw/rapidapi/premier_league/premier_league.csv")
        current_teams = set(league_data['team_name'].unique())
        
        # Get available teams from historical database
        available_teams = predictor.get_available_teams()
        
        # Create team mapping
        team_mapping = create_team_mapping(available_teams, current_teams)
        
        # Validate all teams have valid mappings
        missing_teams = [team for team in current_teams if team not in team_mapping]
        if missing_teams:
            print("\nWarning: Missing mappings for teams:", missing_teams)
            
        # Initialize standings with mapped teams
        standings = {team: {'played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                          'goals_for': 0, 'goals_against': 0, 'points': 0}
                    for team in team_mapping.keys()}
        
        # Get number of matches to simulate
        num_matches = get_num_matches()
        total_matches = num_matches * len(current_teams) // 2
        
        print(f"\nSimulating {total_matches} matches...")
        
        # Simulate matches
        for i in range(total_matches):
            home_idx = (i * 2) % len(current_teams)
            away_idx = (i * 2 + 1) % len(current_teams)
            
            home_team = list(current_teams)[home_idx]
            away_team = list(current_teams)[away_idx]
            
            print(f"Progress: {i+1}/{total_matches} matches simulated")
            print(f"DEBUG: Predicting {home_team} vs {away_team}")
            
            try:
                result = predictor.predict_match(
                    team_mapping[home_team], 
                    team_mapping[away_team]
                )
                update_standings(standings, home_team, away_team, result, league_data)
            except Exception as e:
                print(f"Error predicting {home_team} vs {away_team}: {str(e)}")
        
        # Print final standings
        print_standings(standings)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def get_num_matches():
    """Get number of matches to simulate from user"""
    while True:
        try:
            num = int(input("\nEnter number of games to simulate (1-38): "))
            if 1 <= num <= 38:
                return num
            print("Please enter a number between 1 and 38")
        except ValueError:
            print("Please enter a valid number")

def update_standings(standings: Dict, home_team: str, away_team: str, 
                    result: Dict, league_data: pd.DataFrame) -> None:
    """Update standings with match result"""
    from utils.league_simulator import simulate_match_result
    
    home_goals, away_goals = simulate_match_result(
        home_team, away_team, result, league_data
    )
    
    # Update stats
    standings[home_team]['played'] += 1
    standings[home_team]['goals_for'] += home_goals
    standings[home_team]['goals_against'] += away_goals
    
    standings[away_team]['played'] += 1
    standings[away_team]['goals_for'] += away_goals
    standings[away_team]['goals_against'] += home_goals
    
    if home_goals > away_goals:
        standings[home_team]['wins'] += 1
        standings[home_team]['points'] += 3
        standings[away_team]['losses'] += 1
    elif home_goals < away_goals:
        standings[away_team]['wins'] += 1
        standings[away_team]['points'] += 3
        standings[home_team]['losses'] += 1
    else:
        standings[home_team]['draws'] += 1
        standings[away_team]['draws'] += 1
        standings[home_team]['points'] += 1
        standings[away_team]['points'] += 1

def print_standings(standings: Dict[str, Dict[str, int]]) -> None:
    """Print final league standings"""
    print("\nPredicted Premier League Standings:")
    print("=" * 85)
    print(f"{'Pos':<4} {'Team':<30} {'MP':>4} {'W':>4} {'D':>4} {'L':>4} {'GF':>4} {'GA':>4} {'GD':>4} {'Pts':>6}")
    print("-" * 85)
    
    sorted_teams = sorted(standings.items(), 
                        key=lambda x: (x[1]['points'], x[1]['goals_for'] - x[1]['goals_against']),
                        reverse=True)
    
    for pos, (team, stats) in enumerate(sorted_teams, 1):
        gd = stats['goals_for'] - stats['goals_against']
        print(f"{pos:<4} {team:<30} {stats['played']:>4} {stats['wins']:>4} {stats['draws']:>4} "
              f"{stats['losses']:>4} {stats['goals_for']:>4} {stats['goals_against']:>4} "
              f"{gd:>4} {stats['points']:>6}")

def main():
    # Initialize the model once at startup
    predictor = PredictorSingleton.get_predictor()
    
    while True:
        print("\n=== Soccer Prediction System ===")
        print("1. Predict single match")
        print("2. Predict league standings")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            predict_single_match()
        elif choice == "2":
            predict_league_standings()
        elif choice == "3":
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()