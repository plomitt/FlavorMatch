import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

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


# print("Coefficients:")
# print(model.coef_)
# print("Intercept:")
# print(model.intercept_)



'''
Mean Squared Error (MSE): 0.8607558734932768
Root Mean Squared Error (RMSE): 0.9277692997147926
Mean Absolute Error (MAE): 0.6527063136257795
R² Score: 0.14174072227201126
Explained Variance Score: 0.1418041774213715

Coefficients:
[[ 0.37381254 -0.07561138  0.11793094 -0.05367961  0.00399168 -0.04434787
   0.09655912]
 [ 0.23662484  0.29234352 -0.25055133 -0.14491602  0.00345179 -0.04788253
   0.08281636]
 [ 0.21029632 -0.07718329  0.36793971 -0.11074206  0.01259693 -0.0318721
   0.05895178]
 [-0.09020335 -0.0548819   0.00697875  0.32774002 -0.00443122 -0.00632379
   0.09166575]
 [ 0.0146947  -0.02389778 -0.03577129  0.02786196  0.24621732 -0.01760171
   0.08275184]
 [-0.16614276  0.00056211  0.11744869  0.03092896  0.00354214 -0.01051388
   0.1995933 ]
 [ 0.00172131 -0.05080957  0.04677643 -0.05868874  0.03898328 -0.01688652
   0.37288587]]
Intercept:
[ 5.36040858e-05 -9.48669676e-04  3.15131893e-04 -1.01599637e-03
 -1.06291426e-03  2.80437469e-03  2.60931339e-03] 
'''