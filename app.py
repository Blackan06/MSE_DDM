from flask import Flask, request, jsonify, render_template, send_from_directory
import mlflow.pyfunc
from mlflow.exceptions import MlflowException

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')  

mlflow.set_tracking_uri("mlruns") 

# Load the best model from MLflow Model Registry
model_name = "model"  # Use the registered model name in MLflow
stage = "Production"
model = None

try:
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
    print(f"Successfully loaded model '{model_name}' from stage '{stage}'.")
except MlflowException as e:
    print(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not available"}), 500
    
    data = request.json
    if not data or 'features' not in data:
        return jsonify({"error": "Invalid input"}), 400
    
    # Make prediction
    features = data['features']
    prediction = model.predict([features])
    
    return jsonify({"prediction": int(prediction[0])})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

