from flask import request, jsonify
import joblib
import pandas as pd
import os

MODEL_PATH = 'model/logistic_model.pkl'
SCALER_PATH = 'model/scaler.pkl'

def predict():
    try:
        input_data = request.json

        # Features required for prediction
        features = [
            'Torque(Nm)', 'Hydraulic_Pressure(bar)', 'Cutting(kN)',
            'Coolant_Pressure(bar)', 'Spindle_Speed(RPM)', 'Coolant_Temperature'
        ]

        # Check if all required features are provided
        missing_features = [feature for feature in features if feature not in input_data]
        if missing_features:
            return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

        # Prepare input for prediction
        input_features = {feature: [input_data[feature]] for feature in features}
        input_df = pd.DataFrame(input_features)  # Convert input to DataFrame with feature names

        # Load the trained model and scaler
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # Preprocess the input data (scale it)
        input_scaled = scaler.transform(input_df)

        # Get prediction probabilities
        prediction_prob = model.predict_proba(input_scaled)[0]
        prediction = model.predict(input_scaled)[0]

        # Downtime "Yes" or "No" based on prediction
        downtime = "Yes" if prediction == 1 else "No"
        confidence = prediction_prob[1]  # Probability of class '1' (Downtime = Yes)

        return jsonify({"Downtime": downtime, "Confidence": confidence}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
