# src/database_layer.py
import sqlite3
import pandas as pd
import os

def build_database():
    db_path = "data/spotify_data.db"
    csv_path = "data/raw/spotify_static_data.csv"
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # 1. Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print("Connected to SQLite database.")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Ensure you placed the Kaggle CSV in data/raw/")
        return
        
    # 2. Ingest raw CSV into a base SQL table
    print("Ingesting raw CSV data into SQLite 'raw_tracks' table...")
    df_raw = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df_raw.columns:
        df_raw = df_raw.drop(columns=['Unnamed: 0'])
    if 'track_genre' in df_raw.columns and 'genre' not in df_raw.columns:
        df_raw = df_raw.rename(columns={'track_genre': 'genre'})
        
    df_raw.to_sql("raw_tracks", conn, if_exists="replace", index=False)
    print(f"Loaded {len(df_raw)} raw tracks into SQLite.")
    
    # 3. Clean Data via SQL (Filter out duration anomalies, null targets, and structural bad rows)
    cursor.execute("DROP TABLE IF EXISTS clean_tracks")
    
    clean_sql = """
    CREATE TABLE clean_tracks AS
    SELECT 
        track_id,
        track_name,
        artists AS artist_name,
        album_name,
        popularity AS popularity_score,
        duration_ms,
        CASE WHEN explicit = 1 OR explicit = 'True' OR explicit = 'true' THEN 1 ELSE 0 END AS is_explicit,
        danceability,
        energy,
        key,
        loudness,
        mode,
        speechiness,
        acousticness,
        instrumentalness,
        liveness,
        valence,
        tempo,
        genre
    FROM raw_tracks
    WHERE duration_ms > 30000 
      AND duration_ms < 3600000
      AND popularity_score IS NOT NULL
      AND danceability IS NOT NULL;
    """
    
    cursor.execute(clean_sql)
    conn.commit()
    
    # 4. Verify Database Clean Layer
    clean_count = pd.read_sql_query("SELECT COUNT(*) FROM clean_tracks", conn).iloc[0, 0]
    print(f"Database build complete! Clean table contains {clean_count} records ready for ML modeling.")
    
    conn.close()

if __name__ == "__main__":
    build_database()