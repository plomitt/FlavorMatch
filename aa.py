## Keep only processed info

import pandas as pd

# Load the CSV file
df = pd.read_csv("archive/a_ds.csv")

# Drop original columns
df = df.drop(["original_id", "original"], axis=1)

# Save the df
df.to_csv("archive/aa_ds.csv", index=False)

# Display the df
print(df.head())