from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": float(prediction[0])})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's dynamic port
    app.run(host="0.0.0.0", port=port)         # Bind to all IPs so Render can see i
