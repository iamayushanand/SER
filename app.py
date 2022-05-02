import os
import uuid
from flask import Flask, flash, request, redirect
import pickle
import sklearn
import librosa
import numpy as np
UPLOAD_FOLDER = 'files'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/save-record', methods=['POST'])
def save_record():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    file_name = "audio" + ".wav"
    full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file.save(full_file_name)
    return '<h1>Success</h1>'

@app.route('/predict', methods=['GET'])
def predict():
    model=pickle.load(open("MLP.sav","rb"))
    wave_,sr=librosa.load("files/audio.wav")
    X=[]
    X.append(np.mean(librosa.feature.mfcc(y=wave_,sr=sr).T,axis=0))
    X=np.array(X)
    pred=model.predict(X)

    return '<h1>'+str(pred[0])+'</h1>'

if __name__ == '__main__':
    app.run()
