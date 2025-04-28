"""Probability adjustment utilities"""
from config.model_config import DEFAULT_PROBABILITIES, SMOOTHING_FACTOR

def adjust_match_probabilities(raw_probabilities: dict, home_form: float = None, away_form: float = None) -> dict:
    """
    Adjust raw match probabilities using smoothing factor and team form
    
    Args:
        raw_probabilities (dict): Raw probability predictions
        home_form (float): Home team's recent form (0-1)
        away_form (float): Away team's recent form (0-1)
        
    Returns:
        dict: Adjusted probabilities
    """
    adjusted = {}
    
    # Apply form adjustments if provided
    if home_form is not None and away_form is not None:
        form_diff = home_form - away_form
        form_adjustment = form_diff * 0.1  # Scale form impact
        
        adjusted['home_win'] = raw_probabilities['home_win'] + form_adjustment
        adjusted['away_win'] = raw_probabilities['away_win'] - form_adjustment
        adjusted['draw'] = raw_probabilities['draw']
    else:
        adjusted = raw_probabilities.copy()

    # Apply smoothing using the smoothing factor
    for outcome in adjusted:
        adjusted[outcome] = (adjusted[outcome] * (1 - SMOOTHING_FACTOR) + 
                           DEFAULT_PROBABILITIES[outcome] * SMOOTHING_FACTOR)
    
    # Normalize to ensure probabilities sum to 1
    total = sum(adjusted.values())
    return {k: v/total for k, v in adjusted.items()}