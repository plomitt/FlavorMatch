import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import keras_tuner as kt  # Install using: pip install keras-tuner

# Load dataset
df = pd.read_csv("archive/ds_2.csv")

# Scale features
scaler = StandardScaler()
scaled_arr = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_arr, columns=df.columns)

# Select features and target
X = df_scaled[['processed_ENERC_KCAL', 'processed_PROCNT', 'processed_FAT', 'processed_CHOCDF', 
               'processed_FIBTG', 'processed_flavordb_id', 'processed_cosine_similarity']]
y = df_scaled[['substitution_ENERC_KCAL', 'substitution_PROCNT', 'substitution_FAT', 'substitution_CHOCDF', 
               'substitution_FIBTG', 'substitution_flavordb_id', 'substitution_cosine_similarity']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building function for keras tuner
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(X.shape[1],)))

    # Number of hidden layers (1-3) and neurons per layer (16-128)
    for i in range(hp.Int('num_layers', 1, 3)):  
        model.add(keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=16, max_value=128, step=16),
            activation=hp.Choice('activation', ['relu', 'tanh', 'selu'])
        ))
    
    model.add(keras.layers.Dense(y.shape[1]))

    # Learning rate
    lr = hp.Choice('learning_rate', [0.001, 0.0005, 0.0001])
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model

# Initialize keras tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_mse',
    max_epochs=50,
    factor=3,
    directory='tuner_results',
    project_name='nn_regression_tuning'
)

# Search for the best hyperparameters
tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8, verbose=1)

# Get the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# Train the best model
best_model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)

# Predict and evaluate
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Best Model Mean Squared Error: {mse}")

'''
MSE: 0.7183371782302856
'''