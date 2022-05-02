import pickle
import sklearn
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
st.write("SPEECH EMOTION RECOGNISER")
uploaded_file = st.file_uploader("Choose an audio file")
model=pickle.load(open("MLP.sav","rb"))
scaler=pickle.load(open("scaler.sav","rb"))
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

    fig, ax = plt.subplots(1,2,figsize=(10, 2))
    img1 = librosa.display.waveshow(wave_,sr, ax=ax[0], color='green')
    ax[0].set_title("Waveform")
    ax[0].set_xlabel("")
    ax[0].set_ylabel("")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    n_fft = 2048
    hop_length = 512
    mel_signal = librosa.feature.melspectrogram(y=wave_, sr=sr, hop_length=hop_length,n_fft=n_fft)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    img2 = librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', hop_length=hop_length)
    # fig.colorbar(img2,ax=ax[1],label='dB')
    ax[1].set_title("Mel Spectogram (in dB)")
    ax[1].set_xlabel("")
    ax[1].set_ylabel("")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    fig.tight_layout()
    st.pyplot(fig)


    if st.button("Predict"):
        mfcc=np.mean(librosa.feature.mfcc(wave_).T,axis=0)
        X=[]
        X.append(mfcc)
        X=np.array(X)
        X=scaler.transform(X)
        preds=model.predict(X)
        st.write("The emotion recognized is:")
        st.write(emotions[preds[0]].upper())

