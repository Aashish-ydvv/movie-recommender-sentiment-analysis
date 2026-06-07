Overview:

This project is an interactive web application built with Streamlit that combines recommendation systems with natural language processing. It allows users to explore movies, discover similar titles, and analyze audience reviews in real time.

The application:

Suggests movies based on similarity scores derived from a pre‑computed matrix.
Fetches reviews from TMDB and IMDb.
Classifies reviews as Positive or Negative using a trained LSTM sentiment model.
Presents results in a clean, intuitive interface designed for ease of use.

Features:

Movie Recommendation: 
Select a movie and receive the top five similar titles, complete with posters, ratings, and release years.

Review Sentiment Analysis:
Automatically label reviews with Positive or Negative sentiment using deep learning.

Interactive Interface:
Streamlit tabs provide seamless navigation between recommendations and review analysis.

Reliable Data Integration:
TMDB API supplies movie details, while IMDb scraping ensures reviews are available even when TMDB data is limited.

Dynamic File Handling:
Large files such as similarity.pkl are hosted on Google Drive and downloaded at runtime, making deployment lightweight and cloud‑friendly.

Tech Stack:

Frontend: Streamlit

Backend: Python

Core Libraries: TensorFlow, scikit‑learn, pandas, numpy, requests, BeautifulSoup, gdown

Data Sources: TMDB API, IMDb reviews
