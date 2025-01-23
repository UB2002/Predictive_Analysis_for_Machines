# Machine Downtime Prediction API

This project is a Flask-based RESTful API for predicting machine downtime using a Logistic Regression model. The API allows you to upload a dataset, train the model, and make predictions based on selected features.

---
## project Structure
```bash
machine-downtime-prediction/
│
├── app.py                # Main Flask application file (for initializing the app)
├── routes/               # Folder for routes
│   ├── upload_routes.py  # Route for uploading the file
│   ├── train_routes.py   # Route for training the model
│   └── predict_routes.py # Route for prediction
├── model/                # Folder for saving models and scalers
│   ├── logistic_model.pkl
│   └── scaler.pkl
├── dataset/              # Folder for uploaded datasets
├── requirements.txt      # List of dependencies
└── README.md              # Readme  file

```
---
## Requirements

- Python 3.8 or higher
- Flask
- Scikit-learn
- Pandas
- Joblib

Install dependencies using:
```bash
pip install -r requirements.txt
```
---
## How to Run 

### 1.**Clone the repository:**
    ```bash
       git clone https://github.com/your-username/machine-downtime-prediction.git
    ```
### 2.**Navigate to the project directory:**
    ```bash
       cd machine-downtime-prediction
    ```
### 3. **Install dependencies:**
    ```bash
       pip install -r requirements.txt
    ```
### 4. **Run the Flask server:**
    ```bash
       python app.py
    ```
---
## Features

- **Upload Dataset**: Upload a CSV file containing manufacturing data.
- **Train Model**: Train a Logistic Regression model using selected features.
- **Predict Downtime**: Predict whether a machine will experience downtime based on input features.

---

## Endpoints

### 1. **Upload Dataset**
   - **Endpoint**: `/upload`
   - **Method**: `POST`
   - **Request**:
     - **Body (Form-Data)**:
       - `file`: CSV file containing manufacturing data.
   - **Response**:
     ```json
     {
       "message": "File uploaded successfully",
       "file_path": "dataset/your_dataset.csv"
     }
     ```

### 2. **Train Model**
   - **Endpoint**: `/train`
   - **Method**: `POST`
   - **Request**:
     - **Body (JSON)**:
       ```json
       {
         "file_path": "dataset/your_dataset.csv"
       }
       ```
   - **Response**:
     ```json
     {
       "accuracy": 0.85,
       "f1_score": 0.78
     }
     ```

### 3. **Predict Downtime**
   - **Endpoint**: `/predict`
   - **Method**: `POST`
   - **Request**:
     - **Body (JSON)**:
       ```json
       {
         "Torque(Nm)": 23.38028307,
         "Hydraulic_Pressure(bar)": 124.9278409,
         "Cutting(kN)": 2.63,
         "Coolant_Pressure(bar)": 3.126011472,
         "Spindle_Speed(RPM)": 20419,
         "Coolant_Temperature": 21.1
       }
       ```
   - **Response**:
     ```json
     {
       "Downtime": "No",
       "Confidence": 0.85
     }
     ```

---
