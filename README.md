# Song Recommender System 🎵

A hybrid music recommendation system that combines constraint-based filtering with content-based similarity to suggest songs based on your preferences.

**Live Demo:** [Song Recommender System](https://songrecommendersystem-production.up.railway.app)

---

## Features
- **Content-Based Recommendations:** Find similar songs based on audio features (danceability, energy, tempo, etc.)
- **Constraint Filtering:** Filter by genre, popularity, tempo, acousticness, and more
- **Hybrid Scoring:** Combines similarity, genre matching, and popularity for better results
- **Interactive Web Interface:** Easy-to-use form with real-time recommendations

---

## How It Works
1. **Enter songs you like** (optional) - The system finds similar tracks
2. **Set your preferences** - Filter by genre, tempo, popularity, etc.
3. **Get recommendations** - Receive personalized song suggestions with detailed info

### Recommendation Algorithm
```
User Input → Constraint Filtering → Content Similarity → Hybrid Scoring → Ranked Results
```
- **Constraint Layer:** Hard filters (genre, explicit, popularity ranges)
- **Content Layer:** Cosine similarity on 9 normalized audio features
- **Hybrid Layer:** Weighted scoring with genre boost and popularity

---

## Dataset
Uses a Spotify dataset with 10,000 tracks containing:
- Track metadata (name, artist, album, genre)
- Audio features (danceability, energy, valence, acousticness, etc.)
- Popularity metrics

---

## Tech Stack
- **Backend:** Flask, Python
- **ML Libraries:** scikit-learn, pandas, numpy
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Railway

---

## Project Structure
```
song_recommender_system/
├── app.py                 # Flask application & ML logic
├── requirements.txt       # Python dependencies
├── songs_cleaned.csv       # Spotify dataset of 10k songs
├── templates/
│   └── index.html        # Web interface
└── README.md
```

---

## Contributing
Contributions are welcome! If you want to improve this project:
1. **Fork** the repository.
2. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b my-feature
   ```
3. **Make your changes** and commit them:
   ```bash
   git commit -m "Add my feature"
   ```
4. **Push** to your branch:
   ```bash
   git push origin my-feature
   ```
5. **Open a Pull Request** on the main repository describing your changes.
