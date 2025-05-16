import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_excel("personalized_movie_recommendation_dataset.xlsx")

# Prepare the user-item matrix
user_movie_ratings = df.pivot_table(index="UserID", columns="MovieTitle", values="Rating", aggfunc=np.mean)

# Fill missing values with 0
user_movie_ratings_filled = user_movie_ratings.fillna(0)

# Apply SVD for dimensionality reduction
svd = TruncatedSVD(n_components=20, random_state=42)
latent_matrix = svd.fit_transform(user_movie_ratings_filled)

# Compute the cosine similarity between users
user_similarity = cosine_similarity(latent_matrix)

# Function to get top N movie recommendations for a user
def get_recommendations(user_id, top_n=5):
    user_index = user_id - 1  # Adjust for zero-based indexing
    similarity_scores = user_similarity[user_index]
    similar_users = similarity_scores.argsort()[-top_n-1:-1][::-1]  # Exclude the user themselves

    # Aggregate ratings from similar users
    similar_user_ratings = user_movie_ratings.iloc[similar_users].mean(axis=0)
    unseen_movies = user_movie_ratings.iloc[user_index].isna()
    recommendations = similar_user_ratings[unseen_movies].sort_values(ascending=False).head(top_n)
    return recommendations.index.tolist()

# Example: Recommend movies for a sample user
sample_user = 1
recommended_movies = get_recommendations(sample_user)

# Output the recommendations
print(f"Top Movie Recommendations for User {sample_user}:")
for movie in recommended_movies:
    print(movie)

# Visualizations

# 1. Heatmap: User's movie ratings
plt.figure(figsize=(12, 8))
sns.heatmap(user_movie_ratings_filled, cmap="coolwarm", annot=False, cbar_kws={'label': 'Rating'})
plt.title("User Movie Ratings Heatmap")
plt.xlabel("Movie Title")
plt.ylabel("User ID")
plt.tight_layout()
plt.show()
# 2. Bar Graph: Average rating per genre
genre_rating = df.groupby("Genre")["Rating"].mean().sort_values()
plt.figure(figsize=(10, 6))
sns.barplot(x=genre_rating.values, y=genre_rating.index, palette="viridis")
plt.title("Average Movie Rating by Genre")
plt.xlabel("Average Rating")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

# 3. Bar Graph: Top 5 recommended movies
recommended_movie_titles = recommended_movies
recommended_scores = [user_movie_ratings_filled[movie].mean() for movie in recommended_movie_titles]
plt.figure(figsize=(10, 6))
sns.barplot(x=recommended_movie_titles, y=recommended_scores, palette="magma")
plt.title(f"Top 5 Recommended Movies for User {sample_user}")
plt.xlabel("Movie")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# 4. Box Plot: Distribution of ratings by genre
plt.figure(figsize=(12, 6))
sns.boxplot(x="Genre", y="Rating", data=df, palette="Set2")
plt.title("Rating Distribution by Genre")
plt.xlabel("Genre")
plt.ylabel("Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
