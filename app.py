from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the form as a list
    features = [float(x) for x in request.form.values()]
    features_array = np.array(features).reshape(1, -1)
    # Standardize the features
    features_scaled = scaler.transform(features_array)
    # Make a prediction
    prediction = model.predict(features_scaled)
    
    # Return the result
    return render_template('index.html', prediction_text=f'Predicted Mobile Price: {prediction[0]:.2f} Cartoon Dollars')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
