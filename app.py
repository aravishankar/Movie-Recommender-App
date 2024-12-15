import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from pathlib import Path

from utils import get_popular_movies, myIBCF

DATA_DIR = Path("Data")
BASE_URL = "https://liangfgithub.github.io/MovieData"
IMAGE_URL = "https://liangfgithub.github.io/MovieImages"

@st.cache_data
def load_data():
    try:
        DATA_DIR.mkdir(exist_ok=True)
            
        try:
            movies = requests.get(f"{BASE_URL}/movies.dat?raw=true").content.decode('latin1').split('\n')
            movies = [movie.split("::") for movie in movies if movie]
            movies_df = pd.DataFrame(movies, columns=['MovieID', 'Title', 'Genres'])
            movies_df['MovieID'] = movies_df['MovieID'].astype(int)
            movies_df['image_url'] = movies_df['MovieID'].apply(
                lambda x: f"{IMAGE_URL}/{x}.jpg?raw=true"
            )
        except Exception as e:
            st.error(f"Error loading movies data: {str(e)}")
            st.stop()

        try:
            ratings_df = pd.read_csv(f"{BASE_URL}/ratings.dat?raw=true", 
                                   sep='::', 
                                   engine='python', 
                                   header=None,
                                   names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        except Exception as e:
            st.error(f"Error loading ratings data: {str(e)}")
            st.stop()
        
        try:
            similarity_matrix = pd.read_csv(DATA_DIR / 'similarity_matrix_top30.csv', 
                                         index_col=0)
        except FileNotFoundError:
            st.error("Pre-computed similarity matrix not found in Data directory.")
            st.stop()
        
        try:
            popularity_file = DATA_DIR / 'popularity_rankings.csv'
            if popularity_file.exists():
                popularity_rankings = pd.read_csv(popularity_file)['MovieID_prefix'].values
            else:
                popularity_rankings = get_popular_movies(ratings_df, movies_df)['MovieID_prefix'].values
                pd.DataFrame({'MovieID_prefix': popularity_rankings}).to_csv(popularity_file, index=False)
        except Exception as e:
            st.error(f"Error computing popularity rankings: {str(e)}")
            st.stop()
        
        return movies_df, ratings_df, similarity_matrix, popularity_rankings

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.stop()

st.title('🎬 Movie Recommender System')
st.write("""
Rate movies below from 1-5 stars to see new recommendations! Leave movie ratings at 0 if you haven't seen/don't care to rate them.
""")

with st.spinner('Loading data...'):
    movies_df, ratings_df, similarity_matrix, popularity_rankings = load_data()

@st.cache_data
def get_sample_movies(movies_df, ratings_df, num_rows=20, movies_per_row=6):
    n_movies = num_rows * movies_per_row
    movie_stats = ratings_df.groupby('MovieID')['Rating'].agg(['count', 'mean'])
    top_movies = movie_stats.nlargest(n_movies, 'count')
    sample_movies = movies_df[movies_df['MovieID'].isin(top_movies.index)]
    return sample_movies.head(n_movies)

sample_movies = get_sample_movies(movies_df, ratings_df)

st.write("### Rate Some Movies")

cols_per_row = 6
total_rows = len(sample_movies) // cols_per_row

user_ratings = {}

for row in range(total_rows):
    cols = st.columns(cols_per_row)
    for col in range(cols_per_row):
        idx = row * cols_per_row + col
        movie = sample_movies.iloc[idx]
        with cols[col]:
            st.image(movie['image_url'], width=150)
            st.markdown(f"<p style='text-align: center; font-weight: bold;'>{movie['Title']}</p>", 
                       unsafe_allow_html=True)
            rating = st.select_slider(
                f"Rate {movie['Title']}",
                options=[0, 1, 2, 3, 4, 5],
                value=0,
                key=f"select_{movie['MovieID']}",
                label_visibility="collapsed"
            )
            if rating > 0:
                user_ratings[f"m{movie['MovieID']}"] = rating

if st.button('Get Recommendations', type='primary'):
    if not user_ratings:
        st.warning('Rate at least one movie to get recommendations.')
    else:
        with st.spinner('Generating recommendations...'):
            user_vector = pd.Series(np.nan, index=similarity_matrix.columns)
            for movie_id, rating in user_ratings.items():
                user_vector[movie_id] = rating
            
            recommendations = myIBCF(
                user_vector, 
                similarity_matrix,
                ratings_df,
                movies_df,
                popularity_rankings
            )
            
            st.subheader('Recommended Movies')
            
            rec_cols_per_row = 5
            rec_rows = 2
            
            for row in range(rec_rows):
                cols = st.columns(rec_cols_per_row)
                for col in range(rec_cols_per_row):
                    idx = row * rec_cols_per_row + col
                    if idx < len(recommendations):
                        movie_id = int(recommendations[idx][1:])
                        movie = movies_df[movies_df['MovieID'] == movie_id].iloc[0]
                        with cols[col]:
                            st.image(movie['image_url'], width=150)
                            st.markdown(
                                f"<p style='text-align: center; font-weight: bold;'>{movie['Title']}</p>", 
                                unsafe_allow_html=True
                            )
                            st.markdown(
                                f"<p style='text-align: center;'>Rank {idx + 1}</p>", 
                                unsafe_allow_html=True
                            )

st.markdown("""
---
### How it works
This recommender system uses IBCF to suggest movies you might enjoy based on your ratings. The more movies you rate, the better your recommendations will be.
""")