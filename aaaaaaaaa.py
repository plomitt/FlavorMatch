import pandas as pd

# Load the dataset
df = pd.read_csv("archive/aaaaaaaa_ds.csv")

# List of columns to drop
columns_to_drop = [
    "processed_id", "processed", "processed_category", "processed_label", "processed_weight",
    "substitution", "substitution_id", "substitution_category", "substitution_label", "substitution_weight"
]

# Drop the columns
df = df.drop(columns=columns_to_drop, errors="ignore")

# Save the cleaned dataset
df.to_csv("archive/aaaaaaaaa_ds.csv", index=False)

# Display the first few rows
print(df.head())
