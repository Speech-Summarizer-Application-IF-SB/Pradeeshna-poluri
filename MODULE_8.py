from huggingface_hub import HfApi, HfFolder
import streamlit as st
import os
HF_TOKEN = "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"
os.environ["HF_TOKEN"] = "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"
HfFolder.save_token(HF_TOKEN)
api = HfApi()
try:
    user_info = api.whoami(token=HF_TOKEN)
    st.success(f"logged in as {user_info['name']}")
except Exception as e:
    st.error(f"Token invalid or notfound: {e}")
    st.set_page_config(page_title="Smart Speech Analyzer - Final Dashboard", layout="wide")
HfFolder.save_token(os.getenv("HF_TOKEN"))  # Save token silently
print("Hugging Face token found:", os.getenv("HF_TOKEN"))
import streamlit as st
import pandas as pd
from transformers import pipeline
st.set_page_config(page_title="Smart Speech Analyzer - Final Dashboard", layout="wide")
hf_token = os.getenv("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")
hf_token = "..."

if hf_token:
    st.sidebar.success("Hugging Face token found. Ready to authenticate.")
    if st.sidebar.button("Authenticate with Hugging Face"):
        HfFolder.save_token(hf_token)
        st.sidebar.success("Hugging Face authentication successful.")
else:
    st.sidebar.warning("No Hugging Face token found. Running without authentication.")

st.set_page_config(page_title="Smart Speech Analyzer - Final Dashboard", layout="wide")
st.title("Smart Speech Analyzer – Final Dashboard")
st.write("Analyze, visualize, and download results from your full audio analysis pipeline.")

if os.path.exists("final_results.csv"):
    st.subheader("Previous Results")
    df = pd.read_csv("final_results.csv")
    st.dataframe(df)
else:
    st.warning("No previous results found. Please run a new analysis below.")

st.header("Run New Audio Analysis")

audio_file = st.file_uploader("Upload a .wav file", type=["wav"])
if audio_file is not None:
    file_path = f"uploaded_{audio_file.name}"
    with open(file_path, "wb") as f:
        f.write(audio_file.getbuffer())
    st.info("File uploaded successfully.")

    if st.button("Start Full Analysis"):
        with st.spinner("Processing your audio... Please wait..."):
            try:
                asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-tiny",
                    token=hf_token
                )

                sentiment_pipeline = pipeline("sentiment-analysis")

                emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base"
                )

                asr_result = asr_pipeline(file_path)
                text = asr_result["text"]

                sentiment_result = sentiment_pipeline(text)
                emotion_result = emotion_pipeline(text)

                results = {
                    "Text": text,
                    "Sentiment": sentiment_result[0]["label"],
                    "Sentiment Score": sentiment_result[0]["score"],
                    "Emotion": emotion_result[0]["label"],
                    "Emotion Score": emotion_result[0]["score"]
                }

                df = pd.DataFrame([results])
                df.to_csv("final_results.csv", index=False)

                st.success("Analysis complete. Results saved to final_results.csv")
                st.dataframe(df)

            except Exception as e:
                st.error(f"Error during analysis: {e}")