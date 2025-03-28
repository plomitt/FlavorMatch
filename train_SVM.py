import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("archive/ds_2.csv")

# Scale the features
scaler = StandardScaler()
scaled_arr = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_arr, columns=df.columns)

# Select the feature and target columns
X = df_scaled[['processed_ENERC_KCAL', 'processed_PROCNT', 'processed_FAT', 'processed_CHOCDF', 
        'processed_FIBTG', 'processed_flavordb_id', 'processed_cosine_similarity']]
y = df_scaled[['substitution_ENERC_KCAL', 'substitution_PROCNT', 'substitution_FAT', 'substitution_CHOCDF', 
        'substitution_FIBTG', 'substitution_flavordb_id', 'substitution_cosine_similarity']]


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# Make SVR handle multiple outputs
multi_output_svr = MultiOutputRegressor(svr)

# Train the model
multi_output_svr.fit(X_train, y_train)

# Make predictions
y_pred = multi_output_svr.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")
print(f"Explained Variance Score: {evs}")

'''
Mean Squared Error (MSE): 0.8866638501044166
Root Mean Squared Error (RMSE): 0.9416282972088384
Mean Absolute Error (MAE): 0.5308254861333305
R² Score: 0.11582598680449237
Explained Variance Score: 0.15964264962264213
'''