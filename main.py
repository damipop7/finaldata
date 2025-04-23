# Main
from data_pipeline.ingestion import run_ingestion
from data_pipeline.cleaning import clean_player_data
from utils.log_utils import check_last_update

if __name__ == "__main__":
    # Check and run ingestion if needed
    print(check_last_update())
    
    # Run ingestion if needed
    # run_ingestion()

    # Run data cleaning
    clean_player_data()

