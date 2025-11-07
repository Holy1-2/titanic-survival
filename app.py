from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('titanic_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    pclass = int(request.form['pclass'])
    sex = int(request.form['sex'])
    age = float(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])
    embarked = int(request.form['embarked'])

    # Prepare input for model
    import numpy as np
    import pandas as pd
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    
    # Convert to DataFrame with column names to remove sklearn warning
    columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    input_df = pd.DataFrame(input_data, columns=columns)
    
    # Predict
    prediction = model.predict(input_df)[0]
    result = "🚢 Survived" if prediction == 1 else "💀 Did not survive"

    # Render template with result
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
