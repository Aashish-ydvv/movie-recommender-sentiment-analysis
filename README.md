# Movie Recommender + Review Sentiment Analyzer

An interactive Streamlit app that recommends similar movies and classifies audience reviews as positive or negative using a trained LSTM sentiment model.

**Live demo:** https://movie-recommender-sentiment-analysis-qdr3p8bdm9wgxpv4pqhjsm.streamlit.app/

## Problem

Two common movie-app needs — "what should I watch next" and "is this movie actually well-received" — usually live in separate tools. This project combines a content-based recommender with a review sentiment classifier in a single app, so a user can go from picking a movie to understanding audience reaction to it, without switching tools.

## Architecture

```
User selects a movie
        │
        ├──▶ Recommendation path:
        │        Precomputed similarity matrix (content-based)
        │        → Top 5 similar movies
        │        → TMDB API → posters, ratings, release year
        │
        └──▶ Review analysis path:
                 TMDB API reviews (primary)
                 → IMDb scrape (fallback if TMDB has none)
                 → Tokenizer + LSTM sentiment model
                 → Positive / Negative label per review
                        │
                        ▼
                 Streamlit UI (tabs for each path)
```

## Key Design Decisions

- **Precomputed similarity matrix, loaded lazily:** The similarity matrix is too large to check into git, so it's hosted on Google Drive and downloaded once at runtime (`gdown`), keeping the repo itself lightweight and deployment fast.
- **Fallback review sourcing:** TMDB reviews are tried first; if a movie has none, the app falls back to scraping IMDb reviews, so the sentiment tab still has data to work with for less-reviewed titles.
- **Sentiment as a separate trained model:** Sentiment classification runs through a purpose-trained LSTM (`sentiment_model.h5` + `tokenizer.pkl`) rather than a generic API, keeping inference free and reproducible.
- **Secrets kept out of source:** The TMDB API key is read from Streamlit secrets / environment variables at runtime, never hardcoded — see **Setup** below.

## Tech Stack

**Frontend:** Streamlit
**Backend:** Python
**Core Libraries:** TensorFlow/Keras, scikit-learn, pandas, numpy, requests, BeautifulSoup, gdown
**Data Sources:** TMDB API, IMDb reviews

## Repository Structure

```
movie-recommender-sentiment-analysis/
├── README.md
├── requirements.txt
├── app.py
├── movies.pkl              # movie metadata used for lookups
├── sentiment_model.h5       # trained LSTM sentiment classifier
├── tokenizer.pkl            # tokenizer paired with the sentiment model
└── .streamlit/
    └── secrets.toml.example # template — copy to secrets.toml and fill in your key
```

(`similarity.pkl` is downloaded at runtime from Google Drive — it isn't stored in the repo.)

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Get a free TMDB API key from [themoviedb.org](https://www.themoviedb.org/settings/api).
3. Copy the secrets template and add your key:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   # then edit .streamlit/secrets.toml and paste your key
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

For a deployed app on Streamlit Cloud, add `TMDB_API_KEY` under the app's **Settings → Secrets** instead of committing a secrets file.

## Author

Aashish Deo — Civil Engineer → AI/ML Engineer.
