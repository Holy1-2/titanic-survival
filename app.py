from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
model = joblib.load('titanic_model.pkl')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if data is JSON or form data
        if request.is_json:
            data = request.get_json()
            pclass = int(data['pclass'])
            sex = int(data['sex'])
            age = float(data['age'])
            sibsp = int(data['sibsp'])
            parch = int(data['parch'])
            fare = float(data['fare'])
            embarked = int(data['embarked'])
        else:
            # Get form data
            pclass = int(request.form['pclass'])
            sex = int(request.form['sex'])
            age = float(request.form['age'])
            sibsp = int(request.form['sibsp'])
            parch = int(request.form['parch'])
            fare = float(request.form['fare'])
            embarked = int(request.form['embarked'])

        # Prepare input for model
        input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        
        # Convert to DataFrame with column names to remove sklearn warning
        columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        input_df = pd.DataFrame(input_data, columns=columns)
        
        # Predict
        prediction = model.predict(input_df)[0]
        survived = bool(prediction == 1)

        # Convert numpy types to native Python types for JSON serialization
        response_data = {
            'survived': survived,
            'prediction': int(prediction),  # Convert numpy int to Python int
            'message': 'Survived' if survived else 'Did not survive'
        }

        # Return JSON response for AJAX calls
        if request.is_json:
            return jsonify(response_data)
        else:
            result = "🚢 Survived" if survived else "💀 Did not survive"
            return render_template('index.html', result=result)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        error_response = {
            'error': str(e),
            'message': 'There was an error processing your prediction'
        }
        if request.is_json:
            return jsonify(error_response), 500
        else:
            return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)