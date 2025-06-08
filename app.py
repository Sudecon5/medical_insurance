from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'models/insurance_model.pkl'

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Model file not found at {MODEL_PATH}. Please train the model first.")
    model = None

@app.route('/')
def home():
    """Render the home page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and return prediction"""
    try:
        if model is None:
            return render_template('result.html', 
                                 error="Model not loaded. Please contact administrator.")
        
        # Get form data
        age = float(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']
        
        # Validate inputs
        if not (0 <= age <= 120):
            raise ValueError("Age must be between 0 and 120")
        if not (10 <= bmi <= 60):
            raise ValueError("BMI must be between 10 and 60")
        if not (0 <= children <= 10):
            raise ValueError("Number of children must be between 0 and 10")
        
        # Encode categorical variables
        sex_encoded = 1 if sex == 'male' else 0
        smoker_encoded = 1 if smoker == 'yes' else 0
        
        # Region encoding (one-hot)
        region_northeast = 1 if region == 'northeast' else 0
        region_northwest = 1 if region == 'northwest' else 0
        region_southeast = 1 if region == 'southeast' else 0
        region_southwest = 1 if region == 'southwest' else 0
        
        # Create feature array
        features = np.array([[age, sex_encoded, bmi, children, smoker_encoded,
                             region_northeast, region_northwest, region_southeast, region_southwest]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Format prediction
        predicted_cost = round(prediction, 2)
        
        return render_template('result.html', 
                             prediction=predicted_cost,
                             age=age, sex=sex, bmi=bmi, children=children,
                             smoker=smoker, region=region)
        
    except ValueError as e:
        return render_template('result.html', error=f"Invalid input: {str(e)}")
    except Exception as e:
        return render_template('result.html', error=f"An error occurred: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON)"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        # Extract features
        features = np.array([[
            data['age'],
            1 if data['sex'] == 'male' else 0,
            data['bmi'],
            data['children'],
            1 if data['smoker'] == 'yes' else 0,
            1 if data['region'] == 'northeast' else 0,
            1 if data['region'] == 'northwest' else 0,
            1 if data['region'] == 'southeast' else 0,
            1 if data['region'] == 'southwest' else 0
        ]])
        
        prediction = model.predict(features)[0]
        
        return jsonify({
            'predicted_cost': round(prediction, 2),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # your prediction logic
        return render_template('result.html', prediction=predicted_value)
    except Exception as e:
        return render_template('result.html', error=f"An error occurred: {str(e)}")


if __name__ == '__main__':
    # Use PORT environment variable or default to 10000 for Render.com
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)