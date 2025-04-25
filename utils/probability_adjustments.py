"""Probability adjustment utilities"""
from config.model_config import DEFAULT_PROBABILITIES, SMOOTHING_FACTOR

def adjust_match_probabilities(raw_home: float, raw_draw: float, raw_away: float) -> dict:
    """
    Adjust raw probabilities to be more realistic
    """
    # Configuration
    HOME_WIN_RANGE = (0.35, 0.55)  # More balanced range
    DRAW_RANGE = (0.20, 0.30)      # Typical draw probabilities
    AWAY_WIN_RANGE = (0.25, 0.45)  # More realistic away chances
    
    # Apply constraints
    home_prob = max(HOME_WIN_RANGE[0], min(HOME_WIN_RANGE[1], raw_home))
    away_prob = max(AWAY_WIN_RANGE[0], min(AWAY_WIN_RANGE[1], raw_away))
    draw_prob = max(DRAW_RANGE[0], min(DRAW_RANGE[1], raw_draw))
    
    # Normalize
    total = home_prob + draw_prob + away_prob
    return {
        'home_win': home_prob / total,
        'draw': draw_prob / total,
        'away_win': away_prob / total
    }