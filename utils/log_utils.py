import json
from datetime import datetime
from pathlib import Path

def check_last_update(log_path="data/raw/rapidapi/ingestion_log.json"):
    """
    Check when the data was last updated from the ingestion log
    
    Args:
        log_path (str): Path to the ingestion log file
        
    Returns:
        str: Formatted string with last update time
    """
    try:
        with open(log_path, 'r') as f:
            log_data = json.load(f)
            last_update = datetime.fromisoformat(log_data[-1]['last_update'])
            time_since = datetime.now() - last_update
            
            return (f"Last data update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}\n"
                   f"Time since last update: {time_since.days} days, "
                   f"{time_since.seconds//3600} hours")
    except FileNotFoundError:
        return "No ingestion log found. Run data ingestion first."
    except (json.JSONDecodeError, IndexError, KeyError):
        return "Invalid or corrupted ingestion log file."