import random
import pandas as pd
from typing import Dict, Tuple

def get_team_form_factor(team_name: str, league_data: pd.DataFrame) -> float:
    """Calculate form factor based on current league position and form"""
    try:
        team_row = league_data[league_data['team_name'] == team_name].iloc[0]
        position_factor = 1 - (team_row['rank'] / 20)  # 1 for 1st, 0.05 for 20th
        
        # Convert form (e.g., 'WWDLL') to numeric value
        form_values = {'W': 1, 'D': 0.5, 'L': 0}
        form_factor = sum(form_values[result] for result in team_row['form']) / 5
        
        return (position_factor + form_factor) / 2
    except Exception:
        # Return default form factor for teams without historical data
        return 0.5  # Neutral form

def simulate_match_result(home_team: str, away_team: str, 
                         raw_probs: Dict, league_data: pd.DataFrame) -> Tuple[int, int]:
    """Simulate match result based on probabilities and current form"""
    try:
        # Get form factors with fallback for missing teams
        home_form = get_team_form_factor(home_team, league_data)
        away_form = get_team_form_factor(away_team, league_data)
        
        # Add home advantage
        home_form += 0.1
        
        # Adjust probabilities based on form
        home_prob = raw_probs['home_win'] * (1 + home_form) / 2
        away_prob = raw_probs['away_win'] * (1 + away_form) / 2
        draw_prob = 1 - (home_prob + away_prob)
        
        # Ensure probabilities are valid
        home_prob = max(0.2, min(0.6, home_prob))
        away_prob = max(0.15, min(0.5, away_prob))
        draw_prob = max(0.2, min(0.4, 1 - home_prob - away_prob))
        
        # Normalize probabilities
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total
        
        # Simulate result with realistic scores
        result = random.random()
        if result < home_prob:
            home_goals = max(1, min(4, int(random.gauss(2, 1))))
            away_goals = max(0, min(home_goals - 1, int(random.gauss(1, 1))))
        elif result < home_prob + draw_prob:
            goals = max(0, min(3, int(random.gauss(1.5, 1))))
            home_goals = away_goals = goals
        else:
            away_goals = max(1, min(4, int(random.gauss(2, 1))))
            home_goals = max(0, min(away_goals - 1, int(random.gauss(1, 1))))
            
        return home_goals, away_goals
        
    except Exception as e:
        print(f"Error in match simulation: {str(e)}")
        # Return more realistic fallback result
        return 1, 1  # Return draw with goals