import streamlit as st
import os
from transformers import pipeline
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pyannote.audio import Pipeline
import torch
import warnings

warnings.filterwarnings("ignore")

# --- Helper functions ---

def speaker_diarization(audio_path):
    """Perform speaker diarization and return transcript text"""
    HUGGINGFACE_TOKEN = "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"  # Replace with your actual token
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
    diarization = pipeline(audio_path)
    result = ""
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result += f"[{speaker}]: from {turn.start:.1f}s to {turn.end:.1f}s\n"
    with open("diarized_transcript.txt", "w", encoding="utf-8") as f:
        f.write(result)
    return result

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=120, min_length=40, do_sample=False)[0]['summary_text']
    with open("final_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    return summary

def generate_final_report(diarized_text, summary_text, audio_name):
    report_text = (
        f"=== Final Speech Analysis Report ===\n\n"
        f"Audio File: {audio_name}\n\n"
        f"----- Speaker Diarization Output -----\n\n"
        f"{diarized_text}\n\n"
        f"----- Summarized Report -----\n\n"
        f"{summary_text}\n\n"
    )
    with open("final_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    pdf = canvas.Canvas("final_report.pdf", pagesize=A4)
    width, height = A4
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(100, height - 80, "Final Speech Analysis Report")
    text_object = pdf.beginText(50, height - 120)
    text_object.setFont("Helvetica", 12)
    text_object.textLines(report_text)
    pdf.drawText(text_object)
    pdf.save()
    return "final_report.txt", "final_report.pdf"

# --- Streamlit Interface ---

st.title("Smart Speech Analyzer (Module 6)")
st.write("Upload an audio file and generate a full analysis report with transcription, speaker labeling, and summary.")

uploaded_audio = st.file_uploader("Upload Audio (.wav format only)", type=["wav"])

if uploaded_audio:
    audio_path = os.path.join("uploaded_audio.wav")
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.read())
    st.audio(audio_path, format="audio/wav")

    if st.button("Run Analysis"):
        st.info("Running Speaker Diarization...")
        diarized_text = speaker_diarization(audio_path)

        st.info("Generating Summary...")
        summary_text = summarize_text(diarized_text)

        st.info("Creating Final Report...")
        txt_path, pdf_path = generate_final_report(diarized_text, summary_text, uploaded_audio.name)

        st.success("Analysis complete!")
        st.subheader("Summary Output:")
        st.write(summary_text)

        with open(txt_path, "r", encoding="utf-8") as f:
            st.download_button("Download TXT Report", f, file_name="final_report.txt")

        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="final_report.pdf")
