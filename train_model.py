import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import numpy as np
import json
import re # Import regex for more robust cleaning if needed

print("Loading dataset...")
try:
    data = pd.read_csv("car_data.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'car_data.csv' not found.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# --- Data Preprocessing ---
print("Starting data preprocessing...")

# Standardize column names
data.columns = data.columns.str.replace(' ', '_').str.lower()
print(f"Standardized columns: {data.columns.tolist()}")

# Rename 'selling_price' to 'price'
if 'selling_price' in data.columns:
    data.rename(columns={"selling_price": "price"}, inplace=True)
    print("Renamed 'selling_price' to 'price'.")
elif 'price' not in data.columns:
    print("Error: Target column 'price' (or 'selling_price') not found.")
    exit()

# Drop rows with missing target values ('price')
initial_rows = data.shape[0]
data = data.dropna(subset=["price"])
print(f"Shape after dropping rows with missing Price: {data.shape} (dropped {initial_rows - data.shape[0]} rows)")

# --- Feature Engineering & Cleaning ---

# 1. Extract Manufacturer from 'model' column
if 'model' in data.columns:
    data['manufacturer'] = data['model'].astype(str).str.split().str[0]
    print("Extracted 'manufacturer' feature from 'model'.")
else:
    print("Warning: 'model' column not found. Cannot extract 'manufacturer'.")
    data['manufacturer'] = 'Unknown'

# 2. Clean 'transmission' column
if 'transmission' in data.columns:
    print("Cleaning 'transmission' column...")
    data['transmission'] = data['transmission'].astype(str).str.upper()
    transmission_map = {'MANUAL': 'MANUAL', 'AUTOMATIC': 'AUTOMATIC'}
    data['transmission_cleaned'] = data['transmission'].map(transmission_map)
    data['transmission_cleaned'] = data['transmission_cleaned'].fillna('OTHER')
    print("Unique values in cleaned transmission:", data['transmission_cleaned'].unique())
    data['transmission'] = data['transmission_cleaned']
    data = data.drop(columns=['transmission_cleaned'])
else:
    print("Warning: 'transmission' column not found.")

# 3. Ensure 'car_condition' is numeric (handle potential errors)
if 'car_condition' in data.columns:
    print("Processing 'car_condition' column...")
    data['car_condition'] = pd.to_numeric(data['car_condition'], errors='coerce')
    # Decide how to handle values outside 1-5 range if necessary (e.g., clip or treat as NaN)
    # data['car_condition'] = data['car_condition'].clip(lower=1.0, upper=5.0) # Optional: Clip to range 1-5
    print(f"Car condition range after to_numeric: Min={data['car_condition'].min()}, Max={data['car_condition'].max()}")
else:
    print("Warning: 'car_condition' column not found.")


# --- Feature Selection ---
# Define features to use, including 'car_condition'
features = ["year", "kilometers_driven", "fuel_type", "transmission",
            "owner", "manufacturer", "car_condition"] # Added car_condition

# Verify features exist in the dataframe
available_columns = data.columns.tolist()
features = [f for f in features if f in available_columns] # Only keep features that actually exist
missing_initially = [f for f in ["year", "kilometers_driven", "fuel_type", "transmission", "owner", "manufacturer", "car_condition"] if f not in features]
if missing_initially:
     print(f"Warning: The following features could not be included (missing from data): {missing_initially}")

print(f"Using features: {features}")

# Separate features (X) and target (y)
X = data[features].copy()
y = data["price"]

# Define categorical and numerical columns based on the final features list
categorical_cols = [f for f in ["fuel_type", "transmission", "owner", "manufacturer"] if f in features]
# Added car_condition to numerical
numerical_cols = [f for f in ["year", "kilometers_driven", "car_condition"] if f in features]

# --- Imputation (Fill Missing Values) ---
print("Filling missing values...")
# Fill missing categorical values with "Unknown"
for col in categorical_cols:
    if X[col].isnull().any():
         print(f"Filling missing values in '{col}' with 'Unknown'")
         X[col] = X[col].fillna("Unknown")

# Fill missing numerical values with the median (including car_condition if it had NaNs)
for col in numerical_cols:
    if X[col].isnull().any():
        median_value = X[col].median()
        X[col] = X[col].fillna(median_value)
        print(f"Filled missing values in '{col}' with median: {median_value:.2f}")


# --- Encoding and Scaling ---
print("Encoding categorical features and scaling numerical features...")
encoders = {}
scalers = {}

# Apply Label Encoding to categorical features
for col in categorical_cols:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le
    print(f"Encoded '{col}'. Classes: {le.classes_[:5]}...")

# Apply Standard Scaling to numerical features (now includes car_condition)
if numerical_cols:
    scaler = StandardScaler()
    X.loc[:, numerical_cols] = scaler.fit_transform(X[numerical_cols])
    scalers['numerical'] = scaler
    print(f"Scaled numerical features: {numerical_cols}")

print("Preprocessing complete.")
features_list_ordered = X.columns.tolist()
print(f"Final features for training (order matters): {features_list_ordered}")
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# --- Model Training ---
print("Splitting data and training Random Forest Regressor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete.")

# --- Evaluation ---
print("Evaluating model performance...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
print(f"\nModel Performance (v3):") # Indicate version
print(f"  Train R² Score: {train_r2:.4f}")
print(f"  Test R² Score: {test_r2:.4f}")
print(f"  Test RMSE: {test_rmse:.4f}")

# --- Saving Model, Encoders, Scaler, and Features (v3) ---
print("Saving model v3 artifacts...")
output_dir = "saved_model_rf_v3" # Use v3 directory
os.makedirs(output_dir, exist_ok=True)

# Use v3 filenames
model_path = os.path.join(output_dir, "random_forest_model_v3.pkl")
encoders_path = os.path.join(output_dir, "label_encoders_rf_v3.pkl")
scalers_path = os.path.join(output_dir, "scalers_rf_v3.pkl")
features_path = os.path.join(output_dir, "features_rf_v3.pkl")

joblib.dump(model, model_path)
joblib.dump(encoders, encoders_path)
joblib.dump(scalers, scalers_path)
joblib.dump(features_list_ordered, features_path) # Save the ordered list

print(f"Model saved to: {model_path}")
print(f"Encoders saved to: {encoders_path}")
print(f"Scaler saved to: {scalers_path}")
print(f"Feature list saved to: {features_path}")

# --- Store unique values for dropdowns (v3) ---
unique_values = {}
data_for_unique = data[features].copy()
# Re-apply imputation for unique value generation consistency
for col in categorical_cols:
    if data_for_unique[col].isnull().any():
        data_for_unique[col] = data_for_unique[col].fillna("Unknown")
# Note: car_condition is numerical, not included in unique_values for dropdowns

print("Generating unique values for dropdowns...")
for col in categorical_cols:
    unique_values[col] = sorted(data_for_unique[col].astype(str).unique().tolist())
    print(f"Unique values for '{col}': {unique_values[col][:10]}...")

unique_values_path = os.path.join(output_dir, "unique_categorical_values_rf_v3.json")
with open(unique_values_path, 'w') as f:
    json.dump(unique_values, f, indent=4)
print(f"Unique categorical values saved to: {unique_values_path}")

print("\nTraining script finished.")
