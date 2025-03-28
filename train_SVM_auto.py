import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv("archive/ds_2.csv")

# Scale the features
scaler = StandardScaler()
scaled_arr = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_arr, columns=df.columns)

# Select features and target columns
X = df_scaled[['processed_ENERC_KCAL', 'processed_PROCNT', 'processed_FAT', 'processed_CHOCDF', 
               'processed_FIBTG', 'processed_flavordb_id', 'processed_cosine_similarity']]
y = df_scaled[['substitution_ENERC_KCAL', 'substitution_PROCNT', 'substitution_FAT', 'substitution_CHOCDF', 
               'substitution_FIBTG', 'substitution_flavordb_id', 'substitution_cosine_similarity']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter search space
param_dist = {
    'estimator__C': np.logspace(-3, 3, 10),
    'estimator__epsilon': np.logspace(-3, 1, 10),
    'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'estimator__gamma': ['scale', 'auto']
}

# Initialize the SVR model
base_svr = SVR()

# Make SVR handle multiple outputs
multi_output_svr = MultiOutputRegressor(base_svr)

# Search for best hyperparameters
random_search = RandomizedSearchCV(
    multi_output_svr,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    verbose=1,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    random_state=42
)

# Fit the model
random_search.fit(X_train, y_train)

# Best parameters
print(f"Best hyperparameters: {random_search.best_params_}")

# Train the best model
best_svr = random_search.best_estimator_
best_svr.fit(X_train, y_train)

# Make predictions
y_pred = best_svr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
