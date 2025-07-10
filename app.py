import streamlit as st
import pandas as pd

# Load preprocessed data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Merge data
movie_data = pd.merge(ratings, movies[['movie_id', 'title']], on='movie_id')
movie_stats = movie_data.groupby('title')['rating'].agg(['mean', 'count']).reset_index()
movie_stats.columns = ['title', 'average_rating', 'num_ratings']

# Pivot table for correlation
user_movie_matrix = movie_data.pivot_table(index='user_id', columns='title', values='rating')

# Function to recommend similar movies
def get_recommendations(movie_name, min_ratings=50):
    if movie_name not in user_movie_matrix:
        return pd.DataFrame()
    movie_ratings = user_movie_matrix[movie_name]
    similar = user_movie_matrix.corrwith(movie_ratings)
    corr_df = pd.DataFrame(similar, columns=['correlation'])
    corr_df.dropna(inplace=True)
    corr_df = corr_df.join(movie_stats.set_index('title')[['num_ratings', 'average_rating']])
    results = corr_df[corr_df['num_ratings'] > min_ratings].sort_values(by='correlation', ascending=False)
    return results.head(10)

# Streamlit UI
st.title("🎬 Movie Recommender")

selected_movie = st.selectbox("Choose a movie you like:", sorted(movies['title'].unique()))
min_ratings = st.slider("Minimum number of ratings", 10, 300, 50)

if st.button("Recommend"):
    result_df = get_recommendations(selected_movie, min_ratings)
    if result_df.empty:
        st.warning("No recommendations found.")
    else:
        st.subheader("Recommended Movies:")
        for title, row in result_df.iterrows():
            st.write(f"🎥 **{title}** — ⭐ {row['correlation']:.2f} — Avg Rating: {row['average_rating']:.1f} ({int(row['num_ratings'])} ratings)")

    if not result_df.empty:
        csv = result_df.to_csv(index=True)
        st.download_button("📥 Download Recommendations", csv, "recommendations.csv", "text/csv")
