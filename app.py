from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data["features"]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({"prediction": "yes" if prediction[0] == 1 else "no"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
