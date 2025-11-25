from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load model & encoder
model = pickle.load(open("AirQuality_Model.pkl", "rb"))
encoder = pickle.load(open("AirQuality_LabelEncoder.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Air Quality Flask API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    sensor_value = data["sensor_value"]
    ppm = data["ppm"]

    df = pd.DataFrame([[sensor_value, ppm]],
                      columns=["Sensor_Value", "PPM"])

    pred = model.predict(df)[0]
    label = encoder.inverse_transform([pred])[0]

    return jsonify({
        "sensor_value": sensor_value,
        "ppm": ppm,
        "predicted_quality": label
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
