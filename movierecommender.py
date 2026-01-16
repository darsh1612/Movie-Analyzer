import streamlit as st
import pickle
import pandas as pd

# Load data
movies = pickle.load(open('/Users/darshgupta/Downloads/NLP 2/movies.pkl', 'rb'))
similarity = pickle.load(open('/Users/darshgupta/Downloads/NLP 2/similarity.pkl', 'rb'))

st.set_page_config(page_title="AI Movie Recommender", layout="centered")

st.title("🎬 AI Movie Recommendation System")
st.write("Select a movie and get 5 similar movies using NLP & Machine Learning.")

# Dropdown of movie names
movie_list = movies['title'].values
selected_movie = st.selectbox("Choose a movie", movie_list)

# Recommendation function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

# Button
if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write("👉", movie)
