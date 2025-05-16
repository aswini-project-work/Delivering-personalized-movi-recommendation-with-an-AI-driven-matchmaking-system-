import streamlit as st
import pandas as pd
st.set_page_config(layout="wide")
st.title("🎬 Personalized Movie Recommendation System")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (Excel format)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Sidebar for user selection
    user_ids = df["UserID"].unique()
    selected_user = st.sidebar.selectbox("Select User ID for Recommendations", sorted(user_ids))

    # Prepare user-item matrix
    user_movie_ratings = df.pivot_table(index="UserID", columns="MovieTitle", values="Rating", aggfunc=np.mean)
    user_movie_ratings_filled = user_movie_ratings.fillna(0)

    # Apply SVD and compute similarities
    svd = TruncatedSVD(n_components=20, random_state=42)
    latent_matrix = svd.fit_transform(user_movie_ratings_filled)
    user_similarity = cosine_similarity(latent_matrix)

    # Recommendation function
    def get_recommendations(user_id, top_n=5):
        user_index = user_id - 1
        similarity_scores = user_similarity[user_index]
        similar_users = similarity_scores.argsort()[-top_n-1:-1][::-1]
        similar_user_ratings = user_movie_ratings.iloc[similar_users].mean(axis=0)
        unseen_movies = user_movie_ratings.iloc[user_index].isna()
        recommendations = similar_user_ratings[unseen_movies].sort_values(ascending=False).head(top_n)
        return recommendations.index.tolist()

    # Show recommendations
    recommended_movies = get_recommendations(selected_user)
    st.subheader(f"🎯 Top 5 Movie Recommendations for User {selected_user}")
    for i, movie in enumerate(recommended_movies, start=1):
        st.markdown(f"{i}. **{movie}**")

    # --- Visualizations ---
    st.subheader("📊 Visualizations")

    # 1. Heatmap
    st.markdown("**User-Movie Ratings Heatmap**")
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.heatmap(user_movie_ratings_filled, cmap="coolwarm", ax=ax1, cbar_kws={'label': 'Rating'})
    ax1.set_title("User Movie Ratings Heatmap")
    ax1.set_xlabel("Movie Title")
    ax1.set_ylabel("User ID")
    st.pyplot(fig1)

    # 2. Average rating per genre
    st.markdown("**Average Rating by Genre**")
    genre_rating = df.groupby("Genre")["Rating"].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=genre_rating.values, y=genre_rating.index, ax=ax2, palette="viridis")
    ax2.set_title("Average Movie Rating by Genre")
    ax2.set_xlabel("Average Rating")
    ax2.set_ylabel("Genre")
    st.pyplot(fig2)

    # 3. Bar chart: Top recommended movie scores
    recommended_scores = [user_movie_ratings_filled[movie].mean() for movie in recommended_movies]
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=recommended_movies, y=recommended_scores, ax=ax3, palette="magma")
    ax3.set_title(f"Top 5 Recommended Movies for User {selected_user}")
    ax3.set_xlabel("Movie")
    ax3.set_ylabel("Average Rating")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # 4. Box plot: Ratings by Genre
    st.markdown("**Rating Distribution by Genre**")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="Genre", y="Rating", data=df, ax=ax4, palette="Set2")
    ax4.set_title("Rating Distribution by Genre")
    ax4.set_xlabel("Genre")
    ax4.set_ylabel("Rating")
    plt.xticks(rotation=45)
    st.pyplot(fig4)

else:
    st.warning("Please upload a dataset to proceed.")
