# Ingredient Substitution Prediction

This project explores the effectiveness of various machine learning models in predicting optimal ingredient substitutions based on nutritional content and flavor similarity. It addresses the challenge of identifying suitable ingredient alternatives for individuals with dietary restrictions or allergies while maintaining similar nutritional value and flavor profiles.

This project is for the Machine Learning course of Vrije Universiteit.

## Features

* **Ingredient Parameter Retrieval**: A dataset (`ds_1.csv`) provides nutritional and flavor-related parameters for ingredients based on their name or ID.
* **Substitution Prediction**: Machine learning models predict optimal substitution parameters for a given ingredient.
* **Closest Ingredient Identification**: A utility to find the closest matching ingredient from the dataset based on predicted substitution parameters.
* **Model Training**: Scripts for training and evaluating different regression models.

## Machine Learning Models Evaluated

Four machine learning models were evaluated for this regression task:

* **Linear Regression (LR)**: Used as a baseline model due to its simplicity and interpretability. The optimized version uses Ridge and Lasso Regression with hyperparameter tuning via GridSearchCV.
* **Support Vector Machines (SVM)**: A robust model capable of handling high-dimensional data and capturing complex patterns. Hyperparameter tuning for SVM was attempted but faced excessive runtime issues.
* **Neural Networks (NN)**: A deep learning approach designed to model non-linear relationships between features and targets. Optimized using Keras Tuner for hyperparameter search (number of layers, neurons, activation functions, learning rate).
* **Random Forest Regression (RFR)**: An ensemble learning method that leverages multiple decision trees to improve predictive accuracy and reduce overfitting. Optimized using RandomizedSearchCV for best hyperparameters.

## Dataset

The project utilizes two structured datasets:

* **`ds_1.csv` (Ingredient Feature Dataset)**: Contains 7 nutritional and flavor-related parameters (e.g., energy, protein, fat, carbohydrates, fiber, FlavorDB ID, cosine similarity) for various ingredients. This dataset is used to retrieve ingredient parameters and find the closest ingredient match for a predicted substitution.
* **`ds_2.csv` (Model Training Dataset)**: Comprises 90,283 rows and 14 features, where 7 features describe the original ingredient and 7 target variables describe the substitute. This dataset is used for training the machine learning models.

The dataset creation process involved integrating multiple sources of nutritional, flavor, and substitution data, and processing it through several steps to ensure cleanliness and structure.

## File Structure

* `vu_ml_report.pdf`: The research paper detailing the comparison of SVM, RFR, and ANN models' performance on an ingredient substitution regression task.
* `get_closest_ingredient.py`: A script containing a function to find the closest ingredient in `ds_1.csv` based on a set of substitution features.
* `predict_substitution.py`: A script that loads a trained Random Forest Regressor model and scaler, retrieves ingredient parameters, predicts substitution features, and finds the closest matching ingredient.
* `train_LR_auto.py`: Script for training and hyperparameter tuning of Linear Regression (Ridge and Lasso) models.
* `train_NN_auto.py`: Script for training and hyperparameter tuning of Neural Network models using Keras Tuner.
* `train_RFR_auto.py`: Script for training and hyperparameter tuning of Random Forest Regression models using RandomizedSearchCV.
* `train_SVM_auto.py`: Script for training and hyperparameter tuning of Support Vector Machine models.
* `archive.zip`: Contains all the datasets used for the project.

## Usage

### Predicting Ingredient Substitutions

The `predict_substitution.py` script can be used to predict optimal ingredient substitutions.

```python
# Example usage in predict_substitution.py
ingredient_name = "abalone" # Replace with your desired ingredient
# ... (rest of the script logic)
````

This script will:

1.  Get the ingredient parameters for the specified `ingredient_name`.
2.  Predict the substitution features using the pre-trained RFR model.
3.  Find the closest matching ingredient from `ds_1.csv` based on the predicted substitution features.

### Finding Closest Ingredient

The `get_closest_ingredient.py` script provides a function to find the closest ingredient given a set of features.

```python
# Example usage in get_closest_ingredient.py
predicted_substitution = [157.0,21.9,7.02,0.0,0.0,6511,0.8826214671134949] # Example substitution features
closest_id, closest_name = find_closest_ingredient(predicted_substitution)
print(f"Closest matching ingredient: {closest_name} (ID: {closest_id})")
```

## Training Models

The `train_*.py` scripts are used to train and evaluate the respective machine learning models. Each script handles data loading, scaling, splitting, hyperparameter tuning, and model evaluation.

To train a specific model, run the corresponding script:

```bash
python train_LR_auto.py
python train_NN_auto.py
python train_RFR_auto.py
python train_SVM_auto.py
```

## Results and Performance

The models were assessed based on performance metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R2 score, and Explained Variance Score (EVS). Lower values for MSE, RMSE, and MAE are better, while higher values for R2 score and EVS indicate better performance.

The Random Forest Regression (RFR) model with automatic hyperparameter tuning (RFR auto) outperformed all other models, including the Neural Network (NN), demonstrating the best performance with the lowest MSE and MAE. All auto-tuned models generally outperformed their default counterparts, highlighting the importance of hyperparameter optimization.

The SVM model performed the worst, having the highest MSE and lowest R2 score, indicating it was not suitable for this task.

## Potential Improvements and Future Work

  * **Feature Expansion**: Incorporate additional features such as chemical composition data or detailed physical properties of ingredients to enhance model performance.
  * **Dataset Expansion**: Enlarge the dataset with more ingredient substitution examples to improve model generalization.
  * **Dataset Processing**: Further test the impact of alternative scaling methods and train/test split percentages on training performance.
  * **Advanced Model Architectures**: Explore hybrid models combining ensemble learning with neural networks (e.g., RFR-ANN models) for potentially improved performance.
  * **Explainability Methods**: Utilize techniques like SHAP or feature importance analysis to understand feature contributions and potentially omit less important features for better performance.