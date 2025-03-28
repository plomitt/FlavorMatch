import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# Load the first dataset
df_ingredients = pd.read_csv("archive/ds_1.csv")

# Define the feature columns
feature_cols = [
    "processed_ENERC_KCAL",
    "processed_PROCNT",
    "processed_FAT",
    "processed_CHOCDF",
    "processed_FIBTG",
    "processed_flavordb_id",
    "processed_cosine_similarity"
]

# Convert to a numpy array
ingredient_features = df_ingredients[feature_cols].values

# Function to find the closest ingredient
def find_closest_ingredient(substitution_features):
    substitution_features = np.array(substitution_features).reshape(1, -1)  # Reshape to match dimensions
    distances = cdist(ingredient_features, substitution_features, metric="euclidean")
    closest_idx = np.argmin(distances)  # Get id of the closest match
    
    # Get the ingredient id and name
    closest_id = df_ingredients.iloc[closest_idx]["processed_id"]
    closest_name = df_ingredients.iloc[closest_idx]["processed"]
    
    return closest_id, closest_name

# Example
predicted_substitution = [157.0,21.9,7.02,0.0,0.0,6511,0.8826214671134949]
closest_id, closest_name = find_closest_ingredient(predicted_substitution)

# Output results
print("Closest matching ingredient:")
print(f"ID: {closest_id}")
print(f"Name: {closest_name}")
