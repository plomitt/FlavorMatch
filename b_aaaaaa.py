## Add recipie info

import pandas as pd
import json

# Load the dataset
df = pd.read_csv("archive/aaaaa_ds.csv")

# Load the JSON file
with open("archive/processed_id_recipe1m_map.json", "r") as f:
    recipe_data = json.load(f)

# Initialize new column for recipe_ids
df["processed_recipe_ids"] = "[]"

# Define function to extract and merge recipe data
def extract_recipe_info(row):
    # Get the JSON entry
    info = recipe_data.get(row["processed_id"], {})
    
    # Get recipe_ids or set to empty list if not found
    recipe_ids = info.get("recipe_ids", [])
    
    # Convert recipe_ids to a string
    row["processed_recipe_ids"] = str(recipe_ids) if recipe_ids else "[]"
    
    return row

# Apply the function to each row
df = df.apply(extract_recipe_info, axis=1)

# Save the df
df.to_csv("archive/aaaaaa_ds.csv", index=False)

# Display the df
print(df.head())