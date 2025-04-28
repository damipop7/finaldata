"""Model configuration constants"""

# Match window for calculating recent form
RECENT_MATCHES_WINDOW = 5

# Prediction weights
WEIGHTS = {
    'current_form': 0.4,      # Increased weight for current form
    'historical': 0.2,        # Reduced historical weight  
    'home_advantage': 0.15,   # Adjusted home advantage
    'recent_matches': 0.25    # Weight for recent performance
}

# Default match outcome probabilities
DEFAULT_PROBABILITIES = {
    'home_win': 0.45,
    'draw': 0.25,
    'away_win': 0.30
}

# Smoothing factor for probability adjustments (0-1)
SMOOTHING_FACTOR = 0.3

# Prediction factors flags
PREDICTION_FACTORS = {
    'injuries': True,
    'fatigue': True,
    'motivation': True,
    'home_advantage': True,
    'weather': True
}