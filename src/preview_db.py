# src/preview_db.py
import sqlite3
import pandas as pd

def preview_database():
    db_path = "data/spotify_data.db"
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    print("=== DATABASE TABLE INFO ===")
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print(tables)
    
    print("\n=== CLEAN TRACKS SCHEMA & TYPES ===")
    schema_df = pd.read_sql_query("PRAGMA table_info(clean_tracks);", conn)
    print(schema_df[['name', 'type']])
    
    print("\n=== SAMPLE DATA (First 3 Rows) ===")
    sample_df = pd.read_sql_query("SELECT track_name, artist_name, popularity_score, danceability, energy, genre FROM clean_tracks LIMIT 3", conn)
    print(sample_df)
    
    print("\n=== POPULARITY SCORE STATS ===")
    stats_df = pd.read_sql_query("SELECT MIN(popularity_score) AS min_pop, MAX(popularity_score) AS max_pop, AVG(popularity_score) AS avg_pop FROM clean_tracks", conn)
    print(stats_df)
    
    conn.close()

if __name__ == "__main__":
    preview_database()