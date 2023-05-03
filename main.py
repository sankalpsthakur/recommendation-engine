import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from gensim.models import Word2Vec

# Load the raw data
user_data = pd.read_csv("customers.csv")
item_data = pd.read_csv("itineraries.csv")
interaction_data = pd.read_csv("interaction_data.csv")

# Preprocess the data (cleaning, handling missing values, etc.)
# Implement your own preprocessing steps based on your data

# Extract user features
user_features = user_data[["age", "gender", "location", "travel_preference"]]

# Encode categorical user features (e.g., gender, location, travel_preference)
encoder = OneHotEncoder(sparse=False)
encoded_user_features = encoder.fit_transform(user_features[["gender", "location", "travel_preference"]])

# Combine numerical and encoded categorical features for users
user_features = np.hstack((user_features[["age"]].values, encoded_user_features))

# Extract item features
item_features = item_data[["destination", "duration", "travel_theme", "points_of_interest", "included_meals", "accommodation", "transportation", "price", "activities"]]

# Train a Word2Vec model on the qualitative item features (e.g., destination, travel_theme, points_of_interest, etc.)
qualitative_item_features = ["destination", "travel_theme", "points_of_interest", "included_meals", "accommodation", "transportation", "activities"]
item_feature_sentences = item_data[qualitative_item_features].astype(str).apply(' '.join, axis=1).tolist()

word2vec_model = Word2Vec([sentence.split() for sentence in item_feature_sentences], vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.train(item_feature_sentences, total_examples=len(item_feature_sentences), epochs=10)

# Generate item feature embeddings using the Word2Vec model
item_feature_embeddings = np.array([word2vec_model.wv.get_vector(word) for sentence in item_feature_sentences for word in sentence.split()]).reshape(len(item_feature_sentences), -1)

# Combine numerical (e.g., duration) and vector embeddings for qualitative item features
item_features = np.hstack((item_data[["duration"]].values, item_feature_embeddings))

# Create interaction matrix
interaction_matrix = pd.pivot_table(interaction_data, index="user_id", columns="item_id", values="interaction", fill_value=0)

# Convert the interaction matrix to a sparse format (optional)
interaction_matrix = csr_matrix(interaction_matrix)


import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Hybrid recommendation function
def hybrid_recommendation(user_id, user_features, item_features, interaction_matrix, item_similarity_matrix, k, alpha=0.5):
    # Get user-based collaborative filtering recommendations
    user_interactions = interaction_matrix[user_id]
    user_cf_recommendation_scores = user_interactions.dot(item_similarity_matrix)

    # Get content-based filtering recommendations
    user_feature_vector = user_features[user_id]
    item_feature_similarity_matrix = cosine_similarity(item_features, user_feature_vector.reshape(1, -1))
    user_cb_recommendation_scores = item_feature_similarity_matrix.flatten()

    # Combine the two recommendation scores
    hybrid_scores = alpha * user_cf_recommendation_scores + (1 - alpha) * user_cb_recommendation_scores

    # Get the indices of the top-K recommendations
    top_k_recommendations = np.argsort(hybrid_scores)[::-1][:k]

    return top_k_recommendations

# Reinforcement learning with human feedback
def thompson_sampling(user_id, user_features, item_features, interaction_matrix, item_similarity_matrix, k, num_iterations=1000):
    # Assuming you have a feedback function that returns 1 for positive feedback and 0 for negative feedback
    def get_feedback(user_id, recommended_item_id):
        # Implement the logic to collect human feedback
        pass

    # Initialize the parameters for Thompson Sampling
    item_count = item_features.shape[0]
    item_successes = np.zeros(item_count)
    item_failures = np.zeros(item_count)

    # Thompson Sampling iterations
    for _ in range(num_iterations):
        # Select items using Thompson Sampling
        item_beta_samples = np.random.beta(item_successes + 1, item_failures + 1)
        top_k_items = np.argsort(item_beta_samples)[::-1][:k]

        # Generate hybrid recommendations
        recommendations = hybrid_recommendation(user_id, user_features, item_features, interaction_matrix, item_similarity_matrix, k)

        # Get human feedback for the recommended items
        feedback = [get_feedback(user_id, item_id) for item_id in recommendations]

        # Update item_successes and item_failures based on the feedback
        for i, item_id in enumerate(recommendations):
            if feedback[i] == 1:
                item_successes[item_id] += 1
            else:
                item_failures[item_id] += 1

    # Generate final recommendations based on the updated item_successes and item_failures
    final_recommendations = np.argsort(item_successes / (item_successes + item_failures))[::-1][:k]
    return final_recommendations

# Example usage
user_id = 42
k = 10
alpha = 0.5
num_iterations = 1000

interaction_matrix = csr_matrix(interaction_matrix)
item_similarity_matrix = cosine_similarity(interaction_matrix.T)

# Normalize the user_features and item_features
scaler = MinMaxScaler()
user_features = scaler.fit_transform(user_features)
item_features = scaler.fit_transform(item_features)

final_recommendations = thompson_sampling(user_id, user_features, item_features, interaction_matrix, item_similarity_matrix, k, num_iterations)
print(f"Final recommendations for user {user_id}: {final_recommendations}")


