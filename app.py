import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the trained model using joblib
model_path = "model.pkl"
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to the ML API! Use /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({"error": "No features provided"}), 400

    try:
        features = data['features']
        prediction = model.predict([features])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)