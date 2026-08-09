# src/extract_v1_live_data.py
import os
import time
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

def authenticate_spotify():
    """Loads environment variables and authenticates with Spotify via OAuth."""
    load_dotenv()
    auth_manager = SpotifyOAuth(scope="playlist-read-private")
    return spotipy.Spotify(auth_manager=auth_manager)

def get_tracks_by_search(sp, query="pop", max_tracks=500):
    """Searches for tracks using a standard query string to build the dataset."""
    print(f"Searching Spotify catalog for: '{query}'...")
    
    track_records = []
    limit = 50
    offset = 0
    
    while offset < max_tracks:
        results = sp.search(q=query, type='track', limit=limit, offset=offset)
        tracks = results.get('tracks', {}).get('items', [])
        
        if not tracks:
            break
            
        print(f"Fetched search results {offset + 1} to {offset + len(tracks)}...")
        
        for track in tracks:
            if track is None or track.get('id') is None:
                continue
                
            track_data = {
                'track_id': track['id'],
                'track_name': track['name'],
                'artist_name': track['artists'][0]['name'],
                'popularity_score': track['popularity'],  # Target Variable
                'duration_ms': track['duration_ms'],
                'explicit': track['explicit'],
                'release_date': track.get('album', {}).get('release_date', '')
            }
            track_records.append(track_data)
            
        offset += limit
        time.sleep(1.0)
        
    return pd.DataFrame(track_records)

if __name__ == "__main__":
    sp = authenticate_spotify()
    
    # Use a broad, safe query string
    df = get_tracks_by_search(sp, query="pop", max_tracks=500)
    
    # Save to your raw data folder
    output_path = "data/raw/spotify_live_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nSuccess! Extracted {len(df)} real tracks.")
    print(f"Data saved to {output_path}")