import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

model = joblib.load("models/RFR_best.pkl")
scaler = joblib.load("models/RFR_best_scaler.pkl")
ingredient_name = "abalone"

df_1 = pd.read_csv("archive/ds_1.csv")

feature_cols = [
    "processed_ENERC_KCAL",
    "processed_PROCNT",
    "processed_FAT",
    "processed_CHOCDF",
    "processed_FIBTG",
    "processed_flavordb_id",
    "processed_cosine_similarity"
]

def get_ingredient_params(ingredient_name):
    ingredient_data = df_1[df_1['processed'] == ingredient_name]
    
    if ingredient_data.empty:
        print(f"Ingredient {ingredient_name} not found in the dataset.")
        return None

    ingredient_features = ingredient_data[feature_cols].values
        
    return ingredient_features

def get_substitution_prediction(ingredient_features):
    columns = [
        'processed_ENERC_KCAL', 'processed_PROCNT', 'processed_FAT', 'processed_CHOCDF', 'processed_FIBTG', 'processed_flavordb_id', 'processed_cosine_similarity',
        'substitution_ENERC_KCAL', 'substitution_PROCNT', 'substitution_FAT', 'substitution_CHOCDF', 'substitution_FIBTG', 'substitution_flavordb_id', 'substitution_cosine_similarity'
    ]

    arr = np.array(ingredient_features)
    row = np.hstack([arr[0], np.zeros(7)])
    df1 = pd.DataFrame([row], columns=columns)
    df2 = scaler.transform(df1)
    df3 = pd.DataFrame(df2, columns=columns)
    scaled_features = df3.iloc[:, :7]
    
    predicted_substitution = model.predict(scaled_features)

    arr = np.array(predicted_substitution)
    row = np.hstack([np.zeros(7), arr[0]])
    df1 = pd.DataFrame([row], columns=columns)
    df2 = scaler.inverse_transform(df1)
    df3 = pd.DataFrame(df2, columns=columns)
    inverse_scaled_features = df3.iloc[:, 7:]

    substitution_features = inverse_scaled_features.iloc[0].to_numpy()

    return substitution_features

def find_closest_ingredient(substitution_features):
    ingredient_features = df_1[feature_cols].values
    substitution_features = np.array(substitution_features).reshape(1, -1)
    distances = cdist(ingredient_features, substitution_features, metric="euclidean")
    closest_idx = np.argmin(distances)
    
    closest_id = df_1.iloc[closest_idx]["processed_id"]
    closest_name = df_1.iloc[closest_idx]["processed"]
    
    return closest_id, closest_name


ingredient_features = get_ingredient_params(ingredient_name)

if ingredient_features is not None:
    print('Ingredient features:')
    print(ingredient_features)
    
    substitution_features = get_substitution_prediction(ingredient_features)
    print('Substitution features:')
    print(substitution_features)

    substitution_id, substitution_name = find_closest_ingredient([substitution_features])
    print('Substitution pair:')
    print(f'{ingredient_name} - {substitution_name}')