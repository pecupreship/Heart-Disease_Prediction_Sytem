from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

app = Flask(__name__)

# Load existing model (if it exists)
try:
    with open('hd_model.pkl', 'rb') as f:
        model = pickle.load(f)
except:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            int(request.form['age']),
            int(request.form['sex']),
            int(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            int(request.form['fbs']),
            int(request.form['restecg']),
            float(request.form['thalach']),
            int(request.form['exang']),
            float(request.form['oldpeak']),
            int(request.form['slope']),
            int(request.form['ca']),
            int(request.form['thal'])
        ]

        prediction = model.predict([np.array(features)])
        result = 'Heart Disease Detected 💔' if prediction[0] == 1 else 'No Heart Disease 💖'
        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

@app.route('/add-data', methods=['POST'])
def add_data():
    try:
        values = [
            int(request.form['age']),
            int(request.form['sex']),
            int(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            int(request.form['fbs']),
            int(request.form['restecg']),
            float(request.form['thalach']),
            int(request.form['exang']),
            float(request.form['oldpeak']),
            int(request.form['slope']),
            int(request.form['ca']),
            int(request.form['thal']),
            int(request.form['target'])
        ]

        row = pd.DataFrame([values], columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol',
            'fbs', 'restecg', 'thalach', 'exang',
            'oldpeak', 'slope', 'ca', 'thal', 'target'
        ])

        row.to_csv('heart.csv', mode='a', header=False, index=False)

        # Retrain the model (optional)
        retrain_model()

        return render_template('index.html', prediction_text='✅ New data added and model retrained.')

    except Exception as e:
        return render_template('index.html', prediction_text=f'❌ Error adding data: {e}')

def retrain_model():
    global model
    hd = pd.read_csv('heart.csv')
    X = hd.drop('target', axis=1)
    Y = hd['target']

    model=XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,use_label_encoder=False, eval_metric='mlogloss', subsample=1.0)
    model.fit(X, Y)

    with open('hd_model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    app.run(debug=True)
