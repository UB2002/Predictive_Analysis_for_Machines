import os
from flask import Flask
from routes.upload import upload_data
from routes.train import train_model
from routes.predict import predict

# Initialize Flask app
app = Flask(__name__)

# Paths
UPLOAD_FOLDER = 'dataset'
MODEL_PATH = 'model/logistic_model.pkl'
SCALER_PATH = 'model/scaler.pkl'

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('model', exist_ok=True)


app.add_url_rule('/upload', view_func=upload_data, methods=['POST'])
app.add_url_rule('/train', view_func=train_model, methods=['POST'])
app.add_url_rule('/predict', view_func=predict, methods=['POST'])

@app.route('/', methods=['GET'])
def index():
    return "Hello from the backend!"

# Run the server
if __name__ == "__main__":
    app.run(debug=True)
