# ==========================================
# Task 2: Flask REST API for Order Predictions
# ==========================================

from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Create Flask App

app = Flask(__name__)

# Load Model Once at Startup

model = joblib.load("delivery_model.joblib")

print("Delivery prediction model loaded successfully!")

# Prediction Endpoint

@app.route("/predict", methods=["POST"])
def predict_delivery_time():

    data = request.get_json()

    # Check if JSON exists
    if data is None:
        return jsonify({
            "error": "Request must contain JSON data."
        }), 400

    # Required fields
    required_fields = ["distance_km", "num_items", "rain_flag"]

    missing_fields = []

    for field in required_fields:
        if field not in data:
            missing_fields.append(field)

    if missing_fields:
        return jsonify({
            "error": f"Missing required field(s): {', '.join(missing_fields)}"
        }), 400

    # Prepare input for prediction
    input_df = pd.DataFrame({
        "distance_km": [data["distance_km"]],
        "num_items": [data["num_items"]],
        "rain_flag": [data["rain_flag"]]
    })

    prediction = model.predict(input_df)[0]

    return jsonify({
        "predicted_delivery_time_min": round(float(prediction), 1)
    }), 200

# Run Server

if __name__ == "__main__":
    app.run(debug=True)


