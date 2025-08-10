from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the full pipeline (preprocessing + model)
model = joblib.load("gwp.pkl")  # same file you exported from your notebook

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/pred', methods=['POST'])
def pred():
    try:
        quarter = request.form['quarter']
        department = request.form['department']
        day = request.form['day']
        team = request.form['team']
        targeted_productivity = float(request.form['targeted_productivity'])
        smv = float(request.form['smv'])
        over_time = float(request.form['over_time'])
        incentive = float(request.form['incentive'])
        idle_time = float(request.form['idle_time'])
        idle_men = float(request.form['idle_men'])
        no_of_style_change = int(request.form['no_of_style_change'])
        no_of_workers = int(request.form['no_of_workers'])
        month = request.form['month']

        total = [
                int(quarter), int(department), int(day), int(team),
                float(targeted_productivity), float(smv), float(over_time), float(incentive),
                float(idle_time), int(idle_men), int(no_of_style_change), float(no_of_workers), int(month)
                ]

        columns = [
            'quarter', 'department', 'day', 'team', 'targeted_productivity', 'smv',
            'over_time', 'incentive', 'idle_time', 'idle_men',
            'no_of_style_change', 'no_of_workers', 'month'
        ]

        total_df = pd.DataFrame([total], columns=columns)
        prediction = model.predict(total_df)

        # Interpret prediction
        if prediction <= 0.3:
            text = 'The employee is averagely productive.'
        elif prediction <= 0.8:
            text = 'The employee is medium productive.'
        else:
            text = 'The employee is highly productive.'

        return render_template('submit.html', prediction=text)

    except Exception as e:
        return render_template('submit.html', prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
