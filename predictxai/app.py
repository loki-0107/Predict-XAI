from flask import Flask, request, jsonify
import joblib
import numpy as np
from explain import local_lime_explanation

app = Flask(__name__)

classifier = joblib.load("outputs/failure_classifier.pkl")
regressor = joblib.load("outputs/rul_regressor.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    failure_probability = classifier.predict_proba(features)[0][1]
    failure_prediction = int(failure_probability >= 0.5)
    predicted_rul = float(regressor.predict(features)[0])
    alert = "HIGH RISK" if failure_probability >= 0.7 else "MONITOR"

    return jsonify({
        "failure_probability": round(float(failure_probability), 4),
        "failure_prediction": failure_prediction,
        "predicted_RUL_cycles": round(predicted_rul, 2),
        "maintenance_alert": alert
    })

@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    explanation = local_lime_explanation(
        classifier,
        np.array(data["training_features"]),
        features[0],
        data["feature_names"]
    )
    return jsonify({
        "explanation": explanation.as_list()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)