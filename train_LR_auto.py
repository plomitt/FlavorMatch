import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("archive/ds_2.csv")

# Scale the features
scaler = StandardScaler()
scaled_arr = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_arr, columns=df.columns)

# Select features and target
X = df_scaled[['processed_ENERC_KCAL', 'processed_PROCNT', 'processed_FAT', 'processed_CHOCDF', 
               'processed_FIBTG', 'processed_flavordb_id', 'processed_cosine_similarity']]
y = df_scaled[['substitution_ENERC_KCAL', 'substitution_PROCNT', 'substitution_FAT', 'substitution_CHOCDF', 
               'substitution_FIBTG', 'substitution_flavordb_id', 'substitution_cosine_similarity']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {'alpha': np.logspace(-4, 2, 10)}  # Alpha values from 0.0001 to 100

# Grid search for ridge regression
ridge = Ridge()
ridge_search = GridSearchCV(ridge, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
ridge_search.fit(X_train, y_train)
best_ridge = ridge_search.best_estimator_
print(f"Best ridge alpha: {ridge_search.best_params_}")

# Grid search for lasso regression
lasso = Lasso(max_iter=5000)
lasso_search = GridSearchCV(lasso, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
lasso_search.fit(X_train, y_train)
best_lasso = lasso_search.best_estimator_
print(f"Best lasso alpha: {lasso_search.best_params_}")

# Train final ridge and lasso models
best_ridge.fit(X_train, y_train)
best_lasso.fit(X_train, y_train)

# Make predictions
y_pred_ridge = best_ridge.predict(X_test)
y_pred_lasso = best_lasso.predict(X_test)

# Evaluate the models
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f"Ridge Mean Squared Error: {mse_ridge}")
print(f"Lasso Mean Squared Error: {mse_lasso}")


'''
Fitting 3 folds for each of 10 candidates, totalling 30 fits
Best ridge alpha: {'alpha': 21.54434690031882}
Fitting 3 folds for each of 10 candidates, totalling 30 fits
Best lasso alpha: {'alpha': 0.0001}
Ridge Mean Squared Error: 0.8607540434816144
Lasso Mean Squared Error: 0.8607551729860069
'''