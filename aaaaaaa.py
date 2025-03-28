## Add nutritional values from edamam.json

import pandas as pd
import json

# Load the dataset
df = pd.read_csv("archive/aaaaaa_ds.csv")

# Load the JSON file
with open("archive/edamam.json", "r") as f:
    nutrition_data = json.load(f)

# List of nutrient keys
nutrient_keys = ["ENERC_KCAL", "PROCNT", "FAT", "CHOCDF", "FIBTG"]

# Initialize new columns with -1 to prevent missing values
for nutrient in nutrient_keys:
    df[f"substitution_{nutrient}"] = -1

# Function to extract nutrition data
def extract_nutrition_info(row):
    # Get JSON entry
    info = nutrition_data.get(row["substitution_id"], {})
    
    # Extract values with "substitution_" prefix
    row["substitution_category"] = info.get("category", "-1")
    row["substitution_label"] = info.get("label", "-1")
    row["substitution_weight"] = info.get("weight", -1)
    
    # Extract nutrients
    nutrients = info.get("nutrients", {})
    for nutrient in nutrient_keys:
        row[f"substitution_{nutrient}"] = nutrients.get(nutrient, -1)
    
    return row

# Apply the function to each row
df = df.apply(extract_nutrition_info, axis=1)

# Save the df
df.to_csv("archive/aaaaaaa_ds.csv", index=False)

# Display the df
print(df.head())