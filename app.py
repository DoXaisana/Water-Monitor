from flask import Flask, render_template, jsonify
from database import get_historical_data, get_latest_sensor_data
from predict import predict_next_month

app = Flask(__name__)

@app.route("/")
def index():
    real_data = get_historical_data()
    prediction_data = predict_next_month()
    latest_sensor = get_latest_sensor_data()

    if "error" in prediction_data:
        return f"Error: {prediction_data['error']}"

    return render_template(
        "index.html",
        real_dates=real_data["date"].tolist(),
        real_usage=real_data["daily_usage"].tolist(),
        pred_dates=prediction_data["dates"],
        pred_usage=prediction_data["predicted_usage"],
        total_predicted=prediction_data["total_predicted"],
        latest_sensor=latest_sensor
    )

@app.route("/sensor_data")
def sensor_data():
    """API endpoint to get the latest sensor data."""
    return jsonify(get_latest_sensor_data())

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)