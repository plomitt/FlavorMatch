import pandas as pd
import json


df = pd.read_csv("archive/aaaaa_ds.csv")

# Load the JSON file with substitution pairs
with open("archive/substitution_pairs.json", "r") as f:
    substitutions = json.load(f)

# Convert to df
subs_df = pd.DataFrame(substitutions)

# Rename columns
subs_df = subs_df[["ingredient_processed_id", "substitution", "substitution_processed_id"]]
subs_df.rename(columns={"ingredient_processed_id": "processed_id", "substitution_processed_id": "substitution_id"}, inplace=True)

# Merge into the dataset
merged_df = df.merge(subs_df, on="processed_id", how="left")

# Expand rows with multiple substitutions
expanded_df = merged_df.explode(["substitution", "substitution_id"])

# Save the new dataset
expanded_df.to_csv("archive/aaaaaa_ds.csv", index=False)

# Print
print(expanded_df.head())
