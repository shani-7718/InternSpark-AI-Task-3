from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


@app.route("/")
def home():
    return jsonify({
        "message": "Heart Disease Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array(data["features"]).reshape(1, -1)

    # Apply the same scaling used during training
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    return jsonify({
        "prediction": int(prediction)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)