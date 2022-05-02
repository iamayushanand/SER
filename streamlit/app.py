import pickle
import sklearn
import streamlit as st
import numpy as np
import librosa
st.write("Speech Emotion Recognizer")
uploaded_file = st.file_uploader("Choose an audio file")
model=pickle.load(open("MLP.sav","rb"))
emotions = {
    1 : "neutral",
    2 : "calm",
    3 : "happy",
    4 : "sad",
    5 : "angry",
    6 : "fearful",
    7 : "disgust",
    8 : "surprised"
}

if uploaded_file is not None:
   bytes_data=uploaded_file.getvalue()
   st.write("The uploaded audio file is:")
   st.audio(bytes_data)
   wave_,sr=librosa.load(uploaded_file)
   mfcc=np.mean(librosa.feature.mfcc(wave_).T,axis=0)
   X=[]
   X.append(mfcc)
   X=np.array(X)
   preds=model.predict(X)
   st.write("The emotion recognized is:")
   st.write(emotions[preds[0]])

