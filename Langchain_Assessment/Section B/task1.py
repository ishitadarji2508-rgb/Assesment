# Task 1: Delivery Time Prediction Model


import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

# Create Dummy Delivery Dataset

np.random.seed(10)

records = 120

delivery_data = pd.DataFrame({
    "distance_km": np.random.uniform(1, 25, records),
    "num_items": np.random.randint(1, 8, records),
    "rain_flag": np.random.choice([0, 1], records)
})

# Generate target values
delivery_data["delivery_time_min"] = (
    12
    + delivery_data["distance_km"] * 2.8
    + delivery_data["num_items"] * 2.5
    + delivery_data["rain_flag"] * 7
    + np.random.normal(0, 2, records)
)

print("Sample Dataset:\n")
print(delivery_data.head())

# Prepare Training and Testing Data

features = delivery_data[["distance_km", "num_items", "rain_flag"]]
target = delivery_data["delivery_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=100
)

# Train Linear Regression Model

delivery_model = LinearRegression()
delivery_model.fit(X_train, y_train)

# Evaluate Performance

test_predictions = delivery_model.predict(X_test)

rmse = root_mean_squared_error(y_test, test_predictions)

print(f"\nModel RMSE : {rmse:.2f}")

# Save Model Using Joblib

model_path = "delivery_model.joblib"

joblib.dump(delivery_model, model_path)

size = os.path.getsize(model_path) / 1024

print(f"\nModel saved successfully.")
print(f"File Name : {model_path}")
print(f"Model Size : {size:.2f} KB")

# Reload Saved Model

saved_model = joblib.load(model_path)
# Test Prediction
sample_order = pd.DataFrame({
    "distance_km": [6.5],
    "num_items": [3],
    "rain_flag": [0]
})

prediction_before = delivery_model.predict(sample_order)
prediction_after = saved_model.predict(sample_order)

print("\nPrediction Before Saving :", round(prediction_before[0], 2))
print("Prediction After Loading :", round(prediction_after[0], 2))

# Compare Results
if np.isclose(prediction_before, prediction_after):
    print("\nPASS : Model loaded successfully with identical prediction.")
else:
    print("\nFAIL : Prediction mismatch detected.")