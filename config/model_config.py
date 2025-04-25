"""Model configuration and defaults"""

PREDICTION_FACTORS = {
    'injuries': True,
    'fatigue': True,
    'motivation': True,
    'home_advantage': True,
    'weather': True
}

# Default Premier League probabilities based on historical data
DEFAULT_PROBABILITIES = {
    'home_win': 0.45,  # ~45% home wins
    'draw': 0.25,      # ~25% draws
    'away_win': 0.30   # ~30% away wins
}

# Smoothing factor for probability adjustments (0-1)
# Higher values favor default probabilities
SMOOTHING_FACTOR = 0.3