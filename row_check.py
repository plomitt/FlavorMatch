import pandas as pd

# Load the dataset
df = pd.read_csv("archive/ds_2.csv")

# Get the number of rows
num_rows = df.shape[0]
num_cols = df.shape[1]

# Print the result
print(f"The dataset has {num_rows} rows, {num_cols} cols.")
