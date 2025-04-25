import sqlite3
import pandas as pd

def inspect_database():
    """Inspect the soccer database structure"""
    conn = sqlite3.connect("data/raw/kaggle/soccer/database.sqlite")
    
    # Get all tables
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", 
        conn
    )
    
    print("\nDatabase Tables:")
    print("-" * 20)
    for table in tables['name']:
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 1", conn)
        print(f"\nTable: {table}")
        print("Columns:", ", ".join(df.columns))
    
    # Inspect leagues
    leagues = pd.read_sql_query("SELECT * FROM League", conn)
    print("\nAvailable Leagues:")
    print(leagues)
    
    conn.close()

if __name__ == "__main__":
    inspect_database()