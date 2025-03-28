## Remove duplicates

import pandas as pd

# Load the CSV file
df = pd.read_csv("archive/aa_ds.csv")

# Remove duplicates based on 'processed_id'
df = df.drop_duplicates(subset="processed_id", keep="first")

# Save the dataset
df.to_csv("archive/aaa_ds.csv", index=False)

# Display the df
print(df.head())