from flask import Flask, jsonify

# Initialize Flask app
app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({"message": "Welcome to the ML Breast Cancer Classifier API"})


@app.route("/predict", methods=["POST"])
def predict():
    # Dummy prediction logic
    # Replace with actual model inference
    return jsonify({"prediction": "Malignant"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
