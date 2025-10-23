import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify
import os

class MusicRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                               'instrumentalness', 'tempo', 'speechiness', 
                               'liveness', 'loudness']
        self.scaler = StandardScaler()
        self._prepare_data()
    
    def _prepare_data(self):
        self.df['tempo_normalized'] = self.df['tempo'] / self.df['tempo'].max()
        
        feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 
                       'instrumentalness', 'tempo_normalized', 'speechiness', 
                       'liveness', 'loudness']
        
        self.features_scaled = self.scaler.fit_transform(self.df[feature_cols])
        self.similarity_matrix = cosine_similarity(self.features_scaled)
        
        print(f"Dataset loaded: {len(self.df)} tracks")
        print(f"Genres available: {self.df['genre'].nunique()}")
    
    def get_genres(self):
        return sorted(self.df['genre'].unique().tolist())
    
    def search_tracks(self, query, limit=10):
        matches = self.df[self.df['name'].str.contains(query, case=False, na=False)]
        results = []
        for _, row in matches.head(limit).iterrows():
            results.append({
                'name': row['name'],
                'artists': row['artists'],
                'genre': row['genre']
            })
        return results
    
    def apply_constraints(self, genre=None, explicit=None, min_popularity=0, 
                         max_popularity=100, tempo_min=0, tempo_max=300,
                         min_acousticness=0, max_acousticness=1,
                         min_instrumentalness=0, max_instrumentalness=1,
                         exclude_artists=None):
        mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        if genre:
            if isinstance(genre, str):
                genre = [genre]
            mask &= self.df['genre'].isin(genre)
        
        if explicit is not None:
            mask &= (self.df['explicit'] == explicit)
        
        mask &= (self.df['popularity'] >= min_popularity) & (self.df['popularity'] <= max_popularity)
        mask &= (self.df['tempo'] >= tempo_min) & (self.df['tempo'] <= tempo_max)
        mask &= (self.df['acousticness'] >= min_acousticness) & (self.df['acousticness'] <= max_acousticness)
        mask &= (self.df['instrumentalness'] >= min_instrumentalness) & (self.df['instrumentalness'] <= max_instrumentalness)
        
        if exclude_artists:
            for artist in exclude_artists:
                mask &= ~self.df['artists'].str.contains(artist, case=False, na=False)
        
        return np.where(mask)[0]
    
    def hybrid_recommend(self, seed_tracks=None, n_recommendations=20,
                        genre=None, explicit=None, min_popularity=0,
                        max_popularity=100, tempo_min=0, tempo_max=300,
                        min_acousticness=0, max_acousticness=1,
                        min_instrumentalness=0, max_instrumentalness=1,
                        exclude_artists=None, genre_boost=0.3,
                        popularity_weight=0.2, diversity_factor=0.1):
        valid_indices = self.apply_constraints(
            genre=genre, explicit=explicit, min_popularity=min_popularity,
            max_popularity=max_popularity, tempo_min=tempo_min, tempo_max=tempo_max,
            min_acousticness=min_acousticness, max_acousticness=max_acousticness,
            min_instrumentalness=min_instrumentalness, max_instrumentalness=max_instrumentalness,
            exclude_artists=exclude_artists
        )
        
        if len(valid_indices) == 0:
            return []
        
        if seed_tracks:
            candidate_scores = np.zeros(len(self.df))
            seed_indices = []
            
            for track_name in seed_tracks:
                matches = self.df[self.df['name'].str.contains(track_name, case=False, na=False)]
                if len(matches) > 0:
                    seed_idx = matches.index[0]
                    seed_indices.append(seed_idx)
                    candidate_scores += self.similarity_matrix[seed_idx]
            
            if len(seed_indices) == 0:
                return []
            
            candidate_scores /= len(seed_indices)
            candidate_scores = candidate_scores[valid_indices]
            candidate_indices = valid_indices
            
            mask = ~np.isin(candidate_indices, seed_indices)
            candidate_scores = candidate_scores[mask]
            candidate_indices = candidate_indices[mask]
        else:
            candidate_indices = valid_indices
            candidate_scores = np.ones(len(valid_indices))
        
        if len(candidate_indices) == 0:
            return []
        
        scores = candidate_scores.copy()
        
        if genre and seed_tracks:
            seed_genres = set()
            for track_name in seed_tracks:
                matches = self.df[self.df['name'].str.contains(track_name, case=False, na=False)]
                if len(matches) > 0:
                    seed_genres.add(matches.iloc[0]['genre'])
            
            for i, idx in enumerate(candidate_indices):
                if self.df.iloc[idx]['genre'] in seed_genres:
                    scores[i] += genre_boost
        
        popularity_normalized = self.df.iloc[candidate_indices]['popularity'] / 100
        scores += popularity_normalized.values * popularity_weight
        
        if diversity_factor > 0:
            recommended_so_far = []
            final_indices = []
            final_scores = []
            
            for _ in range(min(n_recommendations, len(candidate_indices))):
                if len(recommended_so_far) > 0:
                    for rec_idx in recommended_so_far:
                        sim_to_recommended = self.similarity_matrix[candidate_indices, rec_idx]
                        scores -= sim_to_recommended * diversity_factor
                
                best_idx = np.argmax(scores)
                final_indices.append(candidate_indices[best_idx])
                final_scores.append(scores[best_idx])
                recommended_so_far.append(candidate_indices[best_idx])
                
                scores = np.delete(scores, best_idx)
                candidate_indices = np.delete(candidate_indices, best_idx)
                
                if len(candidate_indices) == 0:
                    break
        else:
            top_n_idx = np.argsort(scores)[::-1][:n_recommendations]
            final_indices = candidate_indices[top_n_idx]
            final_scores = scores[top_n_idx]
        
        recommendations = []
        for idx, score in zip(final_indices, final_scores):
            row = self.df.iloc[idx]
            recommendations.append({
                'name': row['name'],
                'artists': row['artists'],
                'genre': row['genre'],
                'popularity': int(row['popularity']),
                'tempo': float(row['tempo']),
                'danceability': float(row['danceability']),
                'energy': float(row['energy']),
                'valence': float(row['valence']),
                'acousticness': float(row['acousticness']),
                'instrumentalness': float(row['instrumentalness']),
                'score': float(score)
            })
        
        return recommendations


# Initialize Flask app
app = Flask(__name__)

# Global recommender instance
recommender = None

@app.before_request
def initialize_recommender():
    global recommender
    if recommender is None:
        csv_path = os.environ.get('SPOTIFY_CSV', 'songs_cleaned.csv')
        recommender = MusicRecommender(csv_path)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/genres', methods=['GET'])
def get_genres():
    genres = recommender.get_genres()
    return jsonify({'genres': genres})


@app.route('/api/search', methods=['POST'])
def search_tracks():
    data = request.json
    query = data.get('query', '')
    if len(query) < 2:
        return jsonify({'tracks': []})
    
    tracks = recommender.search_tracks(query, limit=10)
    return jsonify({'tracks': tracks})


@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    
    seed_tracks = data.get('seed_tracks', [])
    if isinstance(seed_tracks, str):
        seed_tracks = [s.strip() for s in seed_tracks.split(',') if s.strip()]
    
    genres = data.get('genres', [])
    if isinstance(genres, str):
        genres = [g.strip() for g in genres.split(',') if g.strip()]
    
    explicit = data.get('explicit')
    if explicit == 'any':
        explicit = None
    elif explicit == 'yes':
        explicit = True
    elif explicit == 'no':
        explicit = False
    
    recommendations = recommender.hybrid_recommend(
        seed_tracks=seed_tracks if seed_tracks else None,
        n_recommendations=int(data.get('n_recommendations', 20)),
        genre=genres if genres else None,
        explicit=explicit,
        min_popularity=int(data.get('min_popularity', 0)),
        max_popularity=int(data.get('max_popularity', 100)),
        tempo_min=float(data.get('tempo_min', 0)),
        tempo_max=float(data.get('tempo_max', 300)),
        min_acousticness=float(data.get('min_acousticness', 0)),
        max_acousticness=float(data.get('max_acousticness', 1)),
        min_instrumentalness=float(data.get('min_instrumentalness', 0)),
        max_instrumentalness=float(data.get('max_instrumentalness', 1))
    )
    
    return jsonify({
        'recommendations': recommendations,
        'count': len(recommendations)
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))