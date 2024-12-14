from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from scipy.spatial.distance import cdist

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

CLIENT_ID = 'ae8d665a1d994c29bde505a2fe60a360'
CLIENT_SECRET = '2bdf1bf93b664db5830f1b6aa7b7fd49'

client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Tải mô hình GMM đã huấn luyện
model = joblib.load('models/model.pkl')

# Tải dữ liệu bài hát
songs_df = pd.read_csv('data/final_data.csv')

# Load scaler đã huấn luyện
scaler = joblib.load('models/scaler.pkl')

def get_song_info(track_id):
    track = sp.track(track_id)
    song_info = {
        'track_name': track['name'],
        'artist_name': ', '.join([artist['name'] for artist in track['artists']]),
        'album_name': track['album']['name'],
        'track_url': track['external_urls']['spotify'],  # URL để nghe bài hát trên Spotify
        'preview_url': track['preview_url'],
        'track_id': track['id'] # URL để nghe đoạn preview (nếu có)
    }
    return song_info

# Hàm đề xuất bài hát
def recommend_songs(user_input):
    genre = user_input.get('genre')

    user_input = np.array(list(user_input.values())).reshape(1, -1)
    user_input_scaled = scaler.transform(user_input)
    
    # Dự đoán cụm của bài hát từ mô hình GMM
    cluster = model.predict(user_input_scaled)
    
    # Lọc các bài hát thuộc cùng một cụm
    cluster_songs = songs_df[(songs_df['cluster'] == cluster[0]) & (songs_df['genre'] == genre)]
    if cluster_songs.empty:
        return []
    
    features = ['genre', 'instrumentalness', 'speechiness', 'acousticness', 'liveness', 'tempo']
    songs_features = cluster_songs[features]
    
    distances = cdist(user_input_scaled, scaler.transform(songs_features), 'euclidean')

    closest_songs_indices = np.argsort(distances[0])[:5]
    closest_songs = cluster_songs.iloc[closest_songs_indices]

    # Lấy thông tin chi tiết của các bài hát
    song_list = []
    for _, song in closest_songs.iterrows():
        track_id = song['track_id']
        song_info = get_song_info(track_id)
        
        if song_info:
            song_list.append(song_info)
    
    return song_list

# API endpoint nhận thông tin từ người dùng
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.json
        recommendations = recommend_songs(user_input)
        if not recommendations:
            return jsonify({"error": "No recommendations found"}), 404
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
