from flask import request, jsonify,Flask
import pandas as pd
import joblib
app = Flask(__name__)
# Load the pre-trained Random Forest model
loaded_model = joblib.load('/content/rf_fraud_detection_model.joblib')
print("Model loaded successfully.")

# Retrieve column names from X_train for consistent feature ordering during inference
model_columns = X_train.columns.tolist()
print(f"Retrieved {len(model_columns)} column names from X_train.")

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Ensure all columns from model_columns are present, fill missing with 0 (for one-hot encoded columns)
        # and align the order
        processed_input = pd.DataFrame(columns=model_columns)
        for col in model_columns:
            if col in input_df.columns:
                processed_input[col] = input_df[col]
            else:
                processed_input[col] = 0 # Assume 0 for missing one-hot encoded features
        
        # Convert boolean columns to int if they exist
        for col in processed_input.select_dtypes(include='bool').columns:
            processed_input[col] = processed_input[col].astype(int)

        # Ensure all columns are numeric, converting if necessary
        for col in processed_input.columns:
            if processed_input[col].dtype == 'object':
                try:
                    processed_input[col] = pd.to_numeric(processed_input[col])
                except ValueError:
                    # Handle cases where non-numeric object columns might remain (e.g., if not one-hot encoded)
                    # For this specific model, all features are expected to be numerical after preprocessing
                    pass

        # Make prediction
        prediction = loaded_model.predict(processed_input[model_columns])

        return jsonify({'prediction': int(prediction[0])})
    else:
        return jsonify({'error': 'Request must be JSON'}), 400

print("Prediction endpoint defined.")
if __name__ == "__main__":
    app.run(debug=True)
