import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from predict_soccer import (
    PredictorSingleton,
    create_team_mapping,
    get_common_words,
    fetch_current_premier_league_teams,
    update_standings
)

def test_team_mapping():
    """Test team name mapping functionality"""
    available_teams = {'Arsenal', 'Chelsea', 'Bournemouth'}
    current_teams = {'Arsenal', 'Chelsea', 'AFC Bournemouth'}
    
    mapping = create_team_mapping(available_teams, current_teams)
    assert mapping['Arsenal'] == 'Arsenal'
    assert mapping['AFC Bournemouth'] == 'Bournemouth'

def test_get_common_words():
    """Test team name word extraction with case insensitivity"""
    test_cases = [
        ('AFC Bournemouth', {'bournemouth'}),
        ('Manchester United', {'manchester'}),
        ('Brighton & Hove Albion', {'brighton', 'hove'}),
        ('Newcastle United', {'newcastle'}),
        ('AFC Wimbledon', {'wimbledon'}),
        # Add more edge cases
        ('West Ham United', {'west', 'ham'}),
        ('Queens Park Rangers', {'queens', 'park', 'rangers'}),
        ('Brighton&Hove', {'brighton', 'hove'})  # Test without spaces around &
    ]
    
    for input_name, expected in test_cases:
        result = get_common_words(input_name)
        assert result == expected, \
            f"Failed for {input_name}: expected {expected}, got {result}"

def test_league_simulation():
    """Test league standings prediction with sample data"""
    # Create sample league data
    league_data = pd.DataFrame({
        'team_name': ['Liverpool', 'Arsenal', 'Chelsea', 'Manchester City'],
        'rank': [1, 2, 3, 4],
        'form': ['WWWDW', 'WDWDW', 'DWWWD', 'WWDLW'],
        'points': [50, 45, 42, 40]
    })
    
    # Mock match result probabilities
    raw_probs = {
        'home_win': 0.45,
        'draw': 0.25,
        'away_win': 0.30
    }
    
    # Test form factor calculation
    from utils.league_simulator import get_team_form_factor
    liverpool_form = get_team_form_factor('Liverpool', league_data)
    assert 0 <= liverpool_form <= 1, "Form factor should be between 0 and 1"
    
    # Test match simulation
    from utils.league_simulator import simulate_match_result
    home_goals, away_goals = simulate_match_result(
        'Liverpool', 'Arsenal', raw_probs, league_data
    )
    assert isinstance(home_goals, int), "Home goals should be integer"
    assert isinstance(away_goals, int), "Away goals should be integer"
    assert 0 <= home_goals <= 5, "Home goals should be between 0 and 5"
    assert 0 <= away_goals <= 5, "Away goals should be between 0 and 5"

    # Test standings update
    standings = {team: {
        'played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
        'goals_for': 0, 'goals_against': 0, 'points': 0
    } for team in league_data['team_name']}
    
    from predict_soccer import update_standings
    with patch('utils.league_simulator.simulate_match_result') as mock_sim:
        mock_sim.return_value = (2, 1)  # Mock a 2-1 home win
        update_standings(standings, 'Liverpool', 'Arsenal', raw_probs, league_data)
        
        # Verify standings update
        assert standings['Liverpool']['wins'] == 1
        assert standings['Liverpool']['points'] == 3
        assert standings['Arsenal']['losses'] == 1
        assert standings['Liverpool']['goals_for'] == 2
        assert standings['Arsenal']['goals_against'] == 2

def test_league_simulation_with_output():
    """Test and display league standings simulation"""
    # Load actual Premier League data
    league_data = pd.read_csv("data/raw/rapidapi/premier_league/premier_league.csv")
    
    # Initialize standings
    teams = league_data['team_name'].unique()
    standings = {team: {
        'played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
        'goals_for': 0, 'goals_against': 0, 'points': 0
    } for team in teams}
    
    # Simulate all matches
    num_teams = len(teams)
    total_matches = num_teams * (num_teams - 1) // 2  # Each team plays each other once
    
    print("\nSimulating Premier League matches...")
    print("-" * 50)
    
    for i in range(total_matches):
        home_idx = (i * 2) % num_teams
        away_idx = (i * 2 + 1) % num_teams
        
        home_team = teams[home_idx]
        away_team = teams[away_idx]
        
        # Use realistic probabilities based on current standings
        raw_probs = {
            'home_win': 0.45,
            'draw': 0.25,
            'away_win': 0.30
        }
        
        try:
            from utils.league_simulator import simulate_match_result
            home_goals, away_goals = simulate_match_result(
                home_team, away_team, raw_probs, league_data
            )
            
            # Update standings
            update_standings(standings, home_team, away_team, raw_probs, league_data)
            
            print(f"Match {i+1}/{total_matches}: {home_team} {home_goals}-{away_goals} {away_team}")
            
        except Exception as e:
            print(f"Error simulating {home_team} vs {away_team}: {e}")
    
    # Print final standings
    print("\nSimulated Premier League Standings")
    print("=" * 85)
    print(f"{'Pos':<4} {'Team':<30} {'MP':>4} {'W':>4} {'D':>4} {'L':>4} "
          f"{'GF':>4} {'GA':>4} {'GD':>4} {'Pts':>6}")
    print("-" * 85)
    
    # Sort teams by points and goal difference
    sorted_teams = sorted(
        standings.items(),
        key=lambda x: (x[1]['points'], x[1]['goals_for'] - x[1]['goals_against']),
        reverse=True
    )
    
    for pos, (team, stats) in enumerate(sorted_teams, 1):
        gd = stats['goals_for'] - stats['goals_against']
        print(f"{pos:<4} {team:<30} {stats['played']:>4} {stats['wins']:>4} "
              f"{stats['draws']:>4} {stats['losses']:>4} {stats['goals_for']:>4} "
              f"{stats['goals_against']:>4} {gd:>4} {stats['points']:>6}")
    
    # Verify simulation integrity
    assert all(s['played'] > 0 for s in standings.values()), "All teams should play matches"
    assert all(s['points'] == s['wins']*3 + s['draws'] for s in standings.values()), "Points calculation error"

if __name__ == '__main__':
    pytest.main(['-v', __file__])