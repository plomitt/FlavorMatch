import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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

# Initialize RandomForestRegressor with the specified hyperparameters
model = RandomForestRegressor(
    n_estimators=200,
    min_samples_split=2,
    min_samples_leaf=4,
    max_features='sqrt',
    max_depth=None,
    bootstrap=False,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

print(pd.DataFrame(y_pred).head())

joblib.dump(model, "RFR_best.pkl")
joblib.dump(scaler, "RFR_best_scaler.pkl")