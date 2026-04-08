from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('models/student_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form
    
    # Initialize feature dict with defaults
    features = {
        'Age': float(data['age']),
        'Study_Hours': float(data['study_hours']),
        'AttendanceRate': float(data['attendance_rate']),
        'GPA': float(data['gpa']),
        'Sleep_Hours': float(data['sleep_hours']),
        'Gender_Male': 1 if data['gender'] == 'Male' else 0,
        'Major_Business': 1 if data['major'] == 'Business' else 0,
        'Major_Education': 1 if data['major'] == 'Education' else 0,
        'Major_Engineering': 1 if data['major'] == 'Engineering' else 0,
        'Major_Science': 1 if data['major'] == 'Science' else 0,
        'PartTimeJob_Yes': 1 if data['part_time_job'] == 'Yes' else 0,
        'ExtraCurricularActivities_Yes': 1 if data['extra_curricular'] == 'Yes' else 0,
        'Grade_B': 1 if data['grade'] == 'B' else 0,
        'Grade_C': 1 if data['grade'] == 'C' else 0,
        'Grade_D': 1 if data['grade'] == 'D' else 0,
    }
    
    # Convert to list in order
    feature_list = [features[col] for col in model.feature_names_in_]
    
    prediction = model.predict([feature_list])[0]
    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)