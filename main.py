from flask import Flask, render_template, request, redirect, url_for
from Algorithms import rf_mean_acc, nn_mean_acc, nb_mean_acc
import pickle
import numpy as np
import os
from markupsafe import Markup

app = Flask(__name__)

# CSS fix by adding modified timestamp
@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

# Load model
model_RF = pickle.load(open('model_RF.pkl','rb'))
model_NN = pickle.load(open('model_NN.pkl','rb'))
model_NB = pickle.load(open('model_NB.pkl','rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/detect")
def detect():
    return render_template('detect.html')

@app.route("/result", methods=['POST','GET'])
def result():
    int_features = [int(x) for x in  request.form.values()]
    final = [np.array(int_features)]
    prediction = model_RF.predict(final)
    print(int_features)
    print(final)
    print(prediction)
    
    if prediction == 1:
        return render_template('result.html', pred='Safe')
    return render_template('result.html', pred='Phishing')


@app.route("/accuracy")
def accuracy():
    return render_template('accuracy.html', nn_mean_acc=nn_mean_acc, rf_mean_acc=rf_mean_acc, nb_mean_acc=nb_mean_acc)