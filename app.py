import joblib
from flask import Flask, request, jsonify

# Load the trained model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Create Flask app
app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request has JSON content
        if not request.is_json:
            return jsonify({'error': 'Request content must be JSON'}), 400

        # Parse the JSON payload
        data = request.get_json()
        features = data['features']

        # Process the input
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
