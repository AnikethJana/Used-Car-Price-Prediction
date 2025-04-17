# Car Price Prediction Model

## Overview

This project implements a machine learning model to predict car prices using a dataset containing car specifications. The model is built using a Random Forest Regressor and includes data preprocessing steps such as handling missing values, encoding categorical features, and scaling numerical features.

## Features

* **Data Loading and Preprocessing:** Loads data from a CSV file, handles missing values, and preprocesses the data for model training.
* **Feature Engineering:** Extracts the manufacturer from the car model.
* **Categorical Feature Encoding:** Encodes categorical features using Label Encoding.
* **Numerical Feature Scaling:** Scales numerical features using StandardScaler.
* **Model Training:** Trains a Random Forest Regressor model.
* **Model Evaluation:** Evaluates the model's performance using R-squared and Root Mean Squared Error (RMSE).
* **Model Persistence:** Saves the trained model, encoders, scalers, and feature list for future use.
* **Unique Value Storage**: Stores unique values for each categorical column in a JSON file, used for generating dropdowns for prediction.

## Technologies Used

* Python
* Pandas
* Scikit-learn (sklearn)
* Joblib
* JSON
* Regular Expressions (re)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/AnikethJana/Used-Car-Price-Prediction.git
    cd Used-Car-Price-Prediction
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install pandas scikit-learn joblib
    ```

## Usage

1.  **Place the dataset:**
    * Ensure the dataset file `car_data.csv` is in the same directory as the Python script.

2.  **Run the script:**

    ```bash
    python train_model.py
    ```

    The script will:
    * Load the dataset.
    * Preprocess the data.
    * Train the Random Forest Regressor model.
    * Evaluate the model's performance.
    * Save the trained model and related files to the `saved_model_rf_v3` directory.
    * Save unique categorical values to `unique_categorical_values_rf_v3.json`.

## Model Artifacts

The following files are saved after the script is executed:

* `saved_model_rf_v3/random_forest_model_v3.pkl`:  The trained Random Forest Regressor model.
* `saved_model_rf_v3/label_encoders_rf_v3.pkl`:  The LabelEncoder objects used for encoding categorical features.
* `saved_model_rf_v3/scalers_rf_v3.pkl`:  The StandardScaler object used for scaling numerical features.
* `saved_model_rf_v3/features_rf_v3.pkl`:  The list of features used for training the model, in the correct order.
* `saved_model_rf_v3/unique_categorical_values_rf_v3.json`: Unique values for each categorical feature.

## Data Source
The dataset used for this project should be named `car_data.csv` and include relevant car features, including a price column.

## Contributing

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Commit your changes.
4.  Push to the branch.
5.  Submit a pull request.

## Author

Aniketh Jana
