"""Team name mapping configuration"""
TEAM_NAME_MAPPING = {
    'AFC Bournemouth': 'Bournemouth',
    'Brighton & Hove Albion': 'Brighton',
    # ...existing mappings...
}

def get_historical_name(current_name: str) -> str:
    return TEAM_NAME_MAPPING.get(current_name, current_name)

def get_current_name(historical_name: str) -> str:
    reverse_mapping = {v: k for k, v in TEAM_NAME_MAPPING.items()}
    return reverse_mapping.get(historical_name, historical_name)