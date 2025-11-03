import streamlit as st
import torch
from transformers import pipeline
import librosa
import numpy as np
import soundfile as sf
from textblob import TextBlob

st.set_page_config(page_title="Emotion and Sentiment Analysis", layout="wide")
st.title("Module 7: Emotion and Sentiment Analysis")

uploaded_audio = st.file_uploader("Upload audio (.wav)", type=["wav"])

if uploaded_audio:
    audio_path = "uploaded_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.read())

    st.audio(audio_path, format="audio/wav")

    st.subheader("Emotion Analysis from Audio")

    y, sr = librosa.load(audio_path, sr=None)
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    if rms > 0.05 and tempo > 100:
        emotion = "Excited / Energetic"
    elif rms < 0.02 and tempo < 80:
        emotion = "Sad / Calm"
    else:
        emotion = "Neutral"

    st.info(f"Detected Emotion: {emotion}")

    st.subheader("Sentiment Analysis from Transcript")

    text_input = st.text_area("Enter or paste transcript text here:")

    if st.button("Analyze Sentiment"):
        if text_input.strip():
            blob = TextBlob(text_input)
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                sentiment = "Positive"
            elif polarity < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            st.success(f"Sentiment: {sentiment}  (Polarity: {round(polarity, 2)})")
        else:
            st.warning("Please enter transcript text for sentiment analysis.")

st.markdown("---")
st.caption("Module 7 - Emotion and Sentiment Analysis Engine")