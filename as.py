import pandas as pd

# Load ratings data
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols)

# Load movie data
movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
               'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')

# Show first few rows
print("Ratings:\n", ratings.head())
print("\nMovies:\n", movies[['movie_id', 'title']].head())

ratings.to_csv('ratings.csv', index=False)
movies.to_csv('movies.csv', index=False)

# Shape of datasets
print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)

# Check for missing values
print("\nMissing values in ratings:\n", ratings.isnull().sum())
print("\nMissing values in movies:\n", movies.isnull().sum())

# Basic statistics
print("\nRatings stats:\n", ratings['rating'].describe())

# Merge ratings and movie titles
movie_data = pd.merge(ratings, movies[['movie_id', 'title']], on='movie_id')

# See the merged data
movie_data.head()

# Group by movie title and calculate mean rating and count of ratings
movie_stats = movie_data.groupby('title')['rating'].agg(['mean', 'count']).reset_index()
movie_stats.columns = ['title', 'average_rating', 'num_ratings']

# Sort by number of ratings (most popular movies)
popular_movies = movie_stats.sort_values(by='num_ratings', ascending=False)

# Show top 10 popular movies
popular_movies.head(10)

import matplotlib.pyplot as plt

# Top 10 most rated movies
top_rated = popular_movies.head(10)

plt.figure(figsize=(10,5))
plt.barh(top_rated['title'], top_rated['num_ratings'], color='skyblue')
plt.xlabel('Number of Ratings')
plt.title('Top 10 Most Rated Movies')
plt.gca().invert_yaxis()
plt.show()

# Create a pivot table (rows = users, columns = movies, values = ratings)
user_movie_matrix = movie_data.pivot_table(index='user_id', columns='title', values='rating')

# Show part of the matrix
user_movie_matrix.head()
# Example: find movies similar to "Toy Story (1995)"
toy_story_ratings = user_movie_matrix['Toy Story (1995)']

# Compute correlation with other movies
similar_to_toy_story = user_movie_matrix.corrwith(toy_story_ratings)

# Create a dataframe for results
corr_toy_story = pd.DataFrame(similar_to_toy_story, columns=['correlation'])
corr_toy_story.dropna(inplace=True)

# Add number of ratings for filtering
corr_toy_story = corr_toy_story.join(movie_stats.set_index('title')['num_ratings'])

# Recommend movies similar to Toy Story, with at least 50 ratings
recommendations = corr_toy_story[corr_toy_story['num_ratings'] > 50].sort_values(by='correlation', ascending=False)

# Show top 10
recommendations.head(10)


