"""
🎬 Movie Recommender + Review Sentiment Analyzer
"""

import os
import pickle
import gdown
import streamlit as st
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# CONFIG
API_KEY = "2524fada1369b424a5d2fe1e31d16d97"
BASE_URL = "https://api.themoviedb.org/3"

# --- CACHED FUNCTION TO DOWNLOAD & LOAD SIMILARITY MATRIX ---
@st.cache_resource
def load_similarity_matrix():
    file_path = "similarity.pkl"
    if not os.path.exists(file_path):
        with st.spinner("Downloading similarity matrix from Google Drive..."):
            file_id = "1zH6zG64jsjezMspRl2DFi1wdIevMfLhR"  # ✅ Your actual file ID
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, file_path, quiet=False)
    return pickle.load(open(file_path, "rb"))

# LOAD OTHER DATA & MODELS
movies = pickle.load(open("movies.pkl", "rb"))
sentiment_model = load_model("sentiment_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

# Load similarity matrix dynamically
similarity = load_similarity_matrix()

# REQUEST SESSION (RETRY ENABLED)
session = requests.Session()
retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

def safe_get(url: str) -> dict:
    try:
        res = session.get(url, timeout=8)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        return {}

# FETCH MOVIE DETAILS
@st.cache_data(show_spinner=False)
def get_movie_details(movie_id: int) -> dict:
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&append_to_response=external_ids"
    data = safe_get(url)
    poster = f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}" if data.get("poster_path") else "https://via.placeholder.com/150"
    imdb_id = data.get("external_ids", {}).get("imdb_id")
    return {
        "poster": poster,
        "imdb_id": imdb_id,
        "title": data.get("title"),
        "release_date": data.get("release_date"),
        "rating": data.get("vote_average")
    }

# RECOMMENDATION FUNCTION
def get_recommendations(movie_title: str) -> list:
    try:
        index = movies[movies["title"] == movie_title].index[0]
    except IndexError:
        return []
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommendations = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].id
        details = get_movie_details(movie_id)
        recommendations.append((movies.iloc[i[0]].title, movie_id, details))
    return recommendations

# REVIEW FETCHING + SENTIMENT
def fetch_tmdb_reviews(movie_id: int, max_reviews: int = 30) -> list:
    url = f"{BASE_URL}/movie/{movie_id}/reviews?api_key={API_KEY}"
    data = safe_get(url)
    return [r.get("content", "") for r in data.get("results", []) if r.get("content")][:max_reviews]

def fetch_imdb_reviews(imdb_id: str, max_reviews: int = 30) -> list:
    if not imdb_id:
        return []
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        page = requests.get(f"https://www.imdb.com/title/{imdb_id}/reviews", headers=headers, timeout=8).text
        soup = BeautifulSoup(page, "lxml")
        selectors = ["div.text.show-more__control", "div.review-container div.text", "div.review-container span"]
        for sel in selectors:
            found = [rev.get_text(strip=True) for rev in soup.select(sel)]
            if found:
                return found[:max_reviews]
        return []
    except Exception as e:
        st.error(f"IMDb fetch failed: {e}")
        return []

def fetch_reviews_with_fallback(movie_id: int, imdb_id: str = None, max_reviews: int = 30) -> list:
    reviews = fetch_tmdb_reviews(movie_id, max_reviews)
    return reviews if reviews else fetch_imdb_reviews(imdb_id, max_reviews)

def predict_sentiment(review: str, maxlen: int = 200) -> str:
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=maxlen)
    pred = sentiment_model.predict(padded, verbose=0)[0][0]
    return "Positive" if pred > 0.5 else "Negative"

# STREAMLIT UI
st.set_page_config(layout="wide")
st.title("🎬 Movie Recommender + Review Sentiment Analyzer")

tab1, tab2 = st.tabs(["📌 Recommendation", "📝 Review Analysis"])

# TAB 1: Recommendation
with tab1:
    st.header("Movie Recommendation")
    movie_title = st.selectbox("Select a movie", movies["title"].values)
    if st.button("Show Recommendation"):
        recommendations = get_recommendations(movie_title)
        if recommendations:
            st.subheader(f"Recommended movies similar to **{movie_title}**:")
            cols = st.columns(5)
            for i, (name, movie_id, details) in enumerate(recommendations):
                with cols[i]:
                    st.image(details["poster"], width=180)
                    st.caption(f"**{name}** — ⭐ {details['rating']} | 📅 {details['release_date'][:4] if details['release_date'] else 'N/A'}")

# TAB 2: Review Analysis
with tab2:
    st.header("Review Sentiment Analysis")
    movie_title = st.selectbox("Select a movie for review analysis", movies["title"].values, key="analysis_movie")
    if st.button("Analyze Reviews"):
        movie_id = movies[movies["title"] == movie_title].iloc[0].id
        details = get_movie_details(movie_id)
        imdb_id = details["imdb_id"]
        reviews = fetch_reviews_with_fallback(movie_id, imdb_id, max_reviews=30)

        if not reviews:
            st.warning("No reviews found on TMDB or IMDb.")
        else:
            review_box = ""
            for r in reviews:
                label = predict_sentiment(r)
                color = "green" if label == "Positive" else "red"
                review_box += f"<p><span style='color:{color}; font-weight:bold'>{label}</span> — {r}</p>"

            st.markdown(
                f"<div style='height:400px; overflow-y:scroll; border:1px solid #ccc; padding:10px'>{review_box}</div>",
                unsafe_allow_html=True
            )
