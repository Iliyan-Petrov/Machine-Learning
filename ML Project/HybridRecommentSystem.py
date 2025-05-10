import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix

# Step 1: Load the MovieLens 100k Dataset
# Load ratings data
ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])

# Load movie data
movies_df = pd.read_csv('ml-100k/u.item', sep='|', encoding='ISO-8859-1', header=None,
                        names=['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'genres'])

# Keep only relevant columns
movies_df = movies_df[['movieId', 'title']]

# Merge the datasets
merged_df = pd.merge(ratings_df, movies_df, on='movieId')

# Step 2: Data Exploration
plt.figure(figsize=(8, 5))
sns.histplot(merged_df['rating'], bins=10, kde=True, color='skyblue')
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# Step 3: Content-Based Filtering Preparation (TF-IDF Vectorizer on Titles)
tfidf = TfidfVectorizer(token_pattern=r'\w+')
tfidf_matrix = tfidf.fit_transform(movies_df['title'])

# Step 4: Collaborative Filtering Preparation (User-Item Interaction Matrix)
user_item_matrix = merged_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
user_item_sparse = csr_matrix(user_item_matrix.values)

# Step 5: Apply SVD for Collaborative Filtering
n_components = 20
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(user_item_sparse)
movie_factors = svd.components_.T

# Step 6: Predict Ratings and Evaluate
predicted_ratings = np.dot(user_factors, movie_factors)
actual_ratings = user_item_matrix.values[user_item_matrix.values > 0]
predicted_ratings_flat = predicted_ratings[user_item_matrix.values > 0]

rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings_flat))
print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
