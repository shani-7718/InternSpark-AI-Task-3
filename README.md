# Heart Disease Prediction API — Task 3

## Project Overview

This project deploys a trained Random Forest machine learning model as a Flask REST API and packages the application using Docker.

The model predicts whether a patient has heart disease based on 13 input features.

## Technologies Used

- Python 3.14
- Flask 3.1.3
- NumPy 2.5.3
- Scikit-learn 1.9.0
- Docker
- Docker Desktop
- WSL 2

## Project Files

- `app.py` — Flask API application
- `heart_disease_model.pkl` — trained Random Forest model
- `scaler.pkl` — feature scaler used during model training
- `requirements.txt` — Python dependencies
- `Dockerfile` — Docker image configuration

## API Endpoints

### Home Endpoint

**GET /**

Returns a message confirming that the API is running.

Example response:

```json
{
  "message": "Heart Disease Prediction API is running"
}
### Prediction Endpoint

**POST /predict**

Accepts 13 patient features and returns a heart disease prediction.

Example request:

```json
{
  "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
}
Example response:

```json
{
  "prediction": 0
}
