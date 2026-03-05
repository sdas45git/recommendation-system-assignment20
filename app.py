import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

df = pd.read_csv("dataset/tmdb_5000_movies.csv")

df['overview'] = df['overview'].fillna("")

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z ]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

df['clean_text'] = df['overview'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(df['clean_text'])

similarity_matrix = cosine_similarity(tfidf_matrix)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend(movie_name, top_n=5):

    idx = indices[movie_name]

    similarity_scores = list(enumerate(similarity_matrix[idx]))

    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similarity_scores = similarity_scores[1:top_n+1]

    movie_indices = [i[0] for i in similarity_scores]

    return df['title'].iloc[movie_indices]

st.title("Movie Recommendation System")

movie_list = df['title'].values

selected_movie = st.selectbox("Select a Movie", movie_list)

if st.button("Recommend"):

    recommendations = recommend(selected_movie)

    st.write("Recommended Movies:")

    for movie in recommendations:
        st.write(movie)