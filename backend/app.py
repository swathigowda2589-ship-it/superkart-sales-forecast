
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Load the trained model (full pipeline: one-hot encoding + tuned XGBoost regressor)
model = joblib.load("superkart_model.joblib")

# Features the model expects, in training order
FEATURES = [
    'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area', 'Product_MRP',
    'Store_Size', 'Store_Location_City_Type', 'Store_Type', 'Product_Id_char',
    'Store_Age_Years', 'Product_Type_Category'
]
NUMERIC_FEATURES = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Age_Years']


def prepare_frame(df):
    """Validate columns / numeric types and return the frame in model column order."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")
    df = df[FEATURES].copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    bad = [c for c in NUMERIC_FEATURES if df[c].isnull().any()]
    if bad:
        raise ValueError(f"Non-numeric or empty value(s) in: {', '.join(bad)}")
    return df


# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"


# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be a JSON object'}), 400

    try:
        # Extract relevant features from the input data and convert into a DataFrame
        input_data = prepare_frame(pd.DataFrame([data]))
        # Make a prediction using the trained model
        prediction = float(model.predict(input_data)[0])
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    # Return the prediction as a JSON response
    return jsonify({'Sales': round(prediction, 2)})


# Define an endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    if 'file' not in request.files:
        return jsonify({'error': "No file uploaded - send the CSV under the 'file' key"}), 400
    file = request.files['file']

    try:
        # Read the file into a DataFrame and validate it
        input_data = prepare_frame(pd.read_csv(file))
        # Make predictions for the batch data
        predictions = model.predict(input_data).tolist()
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {e}'}), 500

    # Create an output dictionary mapping row index to predicted sales
    output_dict = {str(i): round(float(pred), 2) for i, pred in enumerate(predictions)}

    return jsonify(output_dict)


# Run the Flask app in debug mode (local testing only; Docker uses gunicorn)
if __name__ == '__main__':
    superkart_api.run(debug=True, host='0.0.0.0', port=7860)
