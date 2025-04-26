from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

try:
    with open('model/heart_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print("❌ Error loading model:", e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded. Please check again."

    # Get input from form
    data = [
        float(request.form['age']),
        float(request.form['chol']),
        float(request.form['cp']),
        float(request.form['thalach'])
    ]

    prediction = model.predict([data])
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"

    return render_template('index.html', prediction_text=f"Result: {result}")

if __name__ == "__main__":
    app.run(debug=True)
