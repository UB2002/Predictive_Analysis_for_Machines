from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = 'dataset'
MODEL_PATH = 'model/logistic_model.pkl'
model_path = 'model/scaler.pkl'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('model', exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return "hello from the backend"

# File upload route
@app.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return jsonify({"message": "File uploaded successfully", "file_path": file_path}), 200

# Train model route
@app.route('/train', methods=['POST'])
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
        joblib.dump(scaler, model_path)

        return jsonify({"accuracy": accuracy, "f1_score": f1}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Predict route
@app.route('/predict', methods=['POST'])
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
        scaler = joblib.load(model_path)

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

# Run the server
if __name__ == "__main__":
    app.run(debug=True)
