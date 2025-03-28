## Add flavour info

import pandas as pd
import json

# Load the dataset
df = pd.read_csv("archive/aaaaaaa_ds.csv")

# Load the JSON file
with open("archive/ingredient_to_flavordb.json", "r") as f:
    flavor_data = json.load(f)

# List of flavor keys
flavor_keys = ["flavordb_id", "cosine_similarity"]

# Initialize new columns with -1 to prevent missing values
for key in flavor_keys:
    df[f"substitution_{key}"] = -1

# Function to extract data
def extract_flavor_info(row):
    # Get the JSON entry
    info = flavor_data.get(row["substitution_id"], {})
    
    # Extract values
    row["substitution_flavordb_id"] = info.get("flavordb_id", -1)
    row["substitution_cosine_similarity"] = info.get("cosine_similarity", -1)
    
    return row

# Apply the function to each row
df = df.apply(extract_flavor_info, axis=1)

# Save the df
df.to_csv("archive/aaaaaaaa_ds.csv", index=False)

# Display the df
print(df.head())