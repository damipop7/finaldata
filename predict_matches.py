from data_pipeline.modeling import SoccerPredictor
from config.team_mapping import get_historical_name

def main():
    # Initialize predictor
    predictor = SoccerPredictor()
    
    # Train the model - adjust path to your database location
    predictor.train_model("data/raw/kaggle/soccer/database.sqlite")
    
    # Example predictions
    teams_to_predict = [
        ("Manchester United", "Chelsea"),
        ("Arsenal", "Liverpool"),
        ("Manchester City", "Tottenham")
    ]
    
    # Verify team names before prediction
    for home_team, away_team in teams_to_predict:
        historical_home = get_historical_name(home_team)
        historical_away = get_historical_name(away_team)
        print(f"Mapping: {home_team} -> {historical_home}")
        print(f"Mapping: {away_team} -> {historical_away}")
    
    # Make predictions
    for home_team, away_team in teams_to_predict:
        prediction = predictor.predict_match(home_team, away_team)
        print(f"\n{home_team} vs {away_team}:")
        print(f"Win probability: {prediction['win']:.2%}")
        print(f"Draw probability: {prediction['draw']:.2%}")
        print(f"Loss probability: {prediction['loss']:.2%}")
    
    # Predict league standings
    premier_league_teams = [
        "Manchester United", "Chelsea", "Arsenal", "Liverpool",
        "Manchester City", "Tottenham", "Leicester", "West Ham"
    ]
    
    print("\nPredicted Premier League Standings:")
    standings = predictor.predict_league_standings(premier_league_teams)
    for i, team in enumerate(standings, 1):
        print(f"{i}. {team['team']} - {team['predicted_points']:.1f} points")

if __name__ == "__main__":
    main()