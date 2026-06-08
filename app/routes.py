from flask import render_template, request
from app import app
from app.models import predict_with_recommendations

# About page
@app.route('/')
@app.route('/about')
def about():
    return render_template('about.html')

# Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        student_comment = request.form['comment']
        results = predict_with_recommendations(student_comment)
        return render_template('predict.html', comment=student_comment, results=results)
    
    return render_template('predict.html', comment=None, results=None)
