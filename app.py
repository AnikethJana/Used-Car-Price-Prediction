from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import numpy as np
import os
import json

# Initialize Flask app
app = Flask(__name__)

# --- Load Model v3 Components ---
MODEL_DIR = "saved_model_rf_v3" # Use v3 directory
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model_v3.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders_rf_v3.pkl")
SCALERS_PATH = os.path.join(MODEL_DIR, "scalers_rf_v3.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features_rf_v3.pkl")
UNIQUE_VALUES_PATH = os.path.join(MODEL_DIR, "unique_categorical_values_rf_v3.json")

# Check if model files exist
required_files = [MODEL_PATH, ENCODERS_PATH, SCALERS_PATH, FEATURES_PATH, UNIQUE_VALUES_PATH]
if not all(os.path.exists(p) for p in required_files):
    missing = [p for p in required_files if not os.path.exists(p)]
    print(f"Error: Model v3 files not found: {missing}. Please run train_model.py (v3) first.")
    exit()

# Load the components
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    scalers = joblib.load(SCALERS_PATH)
    features_order = joblib.load(FEATURES_PATH)
    with open(UNIQUE_VALUES_PATH, 'r') as f:
        unique_values = json.load(f)
    print("Model v3, encoders, scaler, features list, and unique values loaded successfully.")
    print(f"Feature order expected by model: {features_order}")
except Exception as e:
    print(f"Error loading model v3 components: {e}")
    exit()

# --- Define Routes ---

@app.route('/', methods=['GET'])
def home():
    """Renders the home page with the input form."""
    return render_template('index.html', unique_values=unique_values, prediction_text=None, form_values=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles form submission, preprocesses input, predicts price, and renders result."""
    try:
        # 1. Get data from form
        form_data = request.form.to_dict()
        print(f"Received form data: {form_data}")

        # 2. Create DataFrame from form data
        input_df = pd.DataFrame([form_data])

        # 3. Preprocess the input data
        processed_data = {}

        # Handle categorical features
        categorical_cols_from_encoder = list(encoders.keys())
        print(f"Categorical columns to process: {categorical_cols_from_encoder}")
        for col in categorical_cols_from_encoder:
            if col not in form_data:
                 print(f"Error: Categorical column '{col}' missing in form data.")
                 return render_template('index.html', unique_values=unique_values,
                                        prediction_text=f"Error: Missing input for {col}.", form_values=form_data)
            try:
                value = form_data[col]
                encoder = encoders[col]
                # Handle transmission cleaning consistency
                if col == 'transmission':
                    value = str(value).upper()
                    valid_transmission_values = [str(v).upper() for v in encoder.classes_]
                    if value not in valid_transmission_values:
                         if 'OTHER' in valid_transmission_values: value = 'OTHER'
                         else: raise ValueError(f"Invalid transmission value '{value}'")
                # Handle unseen values for other fields
                elif value not in encoder.classes_:
                    if "Unknown" in encoder.classes_: value = "Unknown"
                    else: raise ValueError(f"Unseen value '{value}' for column '{col}'")

                processed_data[col] = encoder.transform([value])[0]
            except Exception as e:
                 print(f"Error encoding column '{col}' with value '{value}': {e}")
                 return render_template('index.html', unique_values=unique_values,
                                        prediction_text=f"Error processing input for {col}.", form_values=form_data)

        # Handle numerical features (including car_condition)
        numerical_features_for_scaling = []
        if 'numerical' in scalers and hasattr(scalers['numerical'], 'feature_names_in_'):
             numerical_features_for_scaling = scalers['numerical'].feature_names_in_
        elif 'numerical' in scalers: # Fallback
             numerical_features_for_scaling = [f for f in features_order if f not in categorical_cols_from_encoder]
             print(f"Warning: Inferring numerical features for scaling: {numerical_features_for_scaling}")

        numerical_input_values = []
        print(f"Numerical columns to process: {numerical_features_for_scaling}")
        for col in numerical_features_for_scaling:
             if col not in form_data:
                 print(f"Error: Numerical column '{col}' missing in form data.")
                 return render_template('index.html', unique_values=unique_values,
                                        prediction_text=f"Error: Missing input for {col}.", form_values=form_data)
             try:
                 # Get value and convert to float
                 num_value = float(form_data[col])
                 # Optional: Validate car_condition range if needed (though model saw training range)
                 # if col == 'car_condition' and not (1.0 <= num_value <= 5.0):
                 #    print(f"Warning: car_condition '{num_value}' outside expected 1-5 range.")
                 #    # Decide how to handle: clip, error, or allow model to handle
                 #    # num_value = np.clip(num_value, 1.0, 5.0) # Example clipping

                 numerical_input_values.append(num_value)
                 processed_data[col] = num_value # Store pre-scaled
             except ValueError:
                  print(f"Error: Invalid non-numeric value '{form_data[col]}' for numerical column '{col}'.")
                  return render_template('index.html', unique_values=unique_values,
                                        prediction_text=f"Error: Please enter a valid number for {col}.", form_values=form_data)

        # Scale numerical features
        if 'numerical' in scalers and numerical_input_values:
            scaled_numerical_values = scalers['numerical'].transform([numerical_input_values])
            for i, col in enumerate(numerical_features_for_scaling):
                processed_data[col] = scaled_numerical_values[0, i] # Overwrite with scaled
        elif numerical_input_values:
             print("Warning: Numerical scaler not found or no numerical values to scale.")


        # 4. Ensure correct feature order
        final_input_df = pd.DataFrame([processed_data])
        try:
             final_input_df = final_input_df[features_order] # Reorder using loaded list
        except KeyError as e:
             print(f"Error: Column mismatch when ordering features. Missing: {e}")
             return render_template('index.html', unique_values=unique_values,
                                    prediction_text="Error: Internal mismatch in feature processing.", form_values=form_data)


        print(f"Processed data for prediction (ordered): \n{final_input_df.head()}")

        # 5. Make prediction
        prediction = model.predict(final_input_df)
        predicted_price = prediction[0]

        formatted_price = f"₹ {predicted_price:,.2f}"
        print(f"Prediction successful: {formatted_price}")

        # 6. Render the template with the prediction result
        return render_template('index.html',
                               unique_values=unique_values,
                               prediction_text=f"Predicted Car Price: {formatted_price}",
                               form_values=form_data)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        return render_template('index.html',
                               unique_values=unique_values,
                               prediction_text="An error occurred during prediction. Please check logs.",
                               form_values=request.form.to_dict())


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
