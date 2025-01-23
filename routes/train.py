from flask import request, jsonify
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, f1_score
import os

MODEL_PATH = 'model/logistic_model.pkl'
SCALER_PATH = 'model/scaler.pkl'

def train_model():
    try:
        # Load dataset
        file_path = request.json.get('file_path')
        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        data = pd.read_csv(file_path)

        # Drop non-relevant columns
        data = data.drop(columns=['Date', 'Machine_ID'], errors='ignore')

        # Check for missing values and impute them
        imputer = SimpleImputer(strategy='mean')
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

        # Convert categorical target to binary
        data['Downtime'] = data['Downtime'].apply(lambda x: 1 if x == "Machine_Failure" else 0)

        # Select features and target
        features = [
            'Torque(Nm)', 'Hydraulic_Pressure(bar)', 'Cutting(kN)',
            'Coolant_Pressure(bar)', 'Spindle_Speed(RPM)', 'Coolant_Temperature'
        ]
        X = data[features]
        y = data['Downtime']

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Handle class imbalance
        class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

        # Train Logistic Regression model
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_train_scaled, y_train, sample_weight=class_weights)

        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Save model and scaler
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        return jsonify({"accuracy": accuracy, "f1_score": f1}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
