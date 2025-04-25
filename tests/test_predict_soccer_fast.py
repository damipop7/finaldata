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

@pytest.fixture
def sample_data():
    """Fixture providing sample data for tests"""
    return {
        'teams': {
            'available': {'Arsenal', 'Chelsea', 'Liverpool', 'Manchester City'},
            'current': {'Arsenal', 'Chelsea', 'AFC Bournemouth', 'Brighton & Hove Albion'}
        },
        'league_data': pd.DataFrame({
            'team_name': ['Arsenal', 'Chelsea', 'Liverpool'],
            'rank': [1, 2, 3],
            'form': ['WWDWW', 'WDWDW', 'DWWWD'],
            'points': [50, 45, 40]
        }),
        'standings': {
            'Arsenal': {'played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                       'goals_for': 0, 'goals_against': 0, 'points': 0},
            'Chelsea': {'played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                       'goals_for': 0, 'goals_against': 0, 'points': 0}
        }
    }

def test_team_name_processing(sample_data):
    """Test team name processing functions"""
    assert get_common_words('AFC Bournemouth') == {'bournemouth'}
    assert get_common_words('Brighton & Hove Albion') == {'brighton', 'hove'}

def test_team_mapping(sample_data):
    """Test team mapping creation"""
    mapping = create_team_mapping(
        sample_data['teams']['available'],
        sample_data['teams']['current']
    )
    assert mapping['Arsenal'] == 'Arsenal'
    assert mapping['Chelsea'] == 'Chelsea'
    assert mapping['AFC Bournemouth'] == 'Bournemouth'

@patch('pandas.read_csv')
def test_fetch_teams(mock_read_csv, sample_data):
    """Test fetching current teams from CSV"""
    mock_read_csv.return_value = sample_data['league_data']
    teams = fetch_current_premier_league_teams()
    assert isinstance(teams, set)
    assert 'Arsenal' in teams
    assert 'Chelsea' in teams

@patch('utils.league_simulator.simulate_match_result')
def test_update_standings(mock_simulate, sample_data):
    """Test standings update with mocked match simulation"""
    mock_simulate.return_value = (2, 1)  # Home win scenario
    
    standings = sample_data['standings'].copy()
    result = {'home_win': 0.6, 'draw': 0.2, 'away_win': 0.2}
    
    update_standings(standings, 'Arsenal', 'Chelsea', result, sample_data['league_data'])
    
    assert standings['Arsenal']['wins'] == 1
    assert standings['Arsenal']['points'] == 3
    assert standings['Chelsea']['losses'] == 1
    assert standings['Arsenal']['goals_for'] == 2

def test_predictor_singleton():
    """Test PredictorSingleton pattern"""
    with patch('predict_soccer.SoccerPredictor') as MockPredictor:
        # First call should create instance
        predictor1 = PredictorSingleton.get_predictor()
        assert MockPredictor.called
        
        # Second call should return same instance
        predictor2 = PredictorSingleton.get_predictor()
        assert predictor1 is predictor2
        assert MockPredictor.call_count == 1

if __name__ == '__main__':
    pytest.main(['-v', __file__])