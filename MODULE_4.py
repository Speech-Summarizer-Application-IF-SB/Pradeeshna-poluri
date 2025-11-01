import os
from transformers import pipeline

def summarize_text(text, max_length=150, min_length=50):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )[0]['summary_text']
    return summary

def load_diarized_transcript(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcript file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def speaker_wise_summary(transcript_text):
    speakers = {}
    for line in transcript_text.split("\n"):
        if line.strip():
            if line.startswith("[Speaker"):
                speaker, content = line.split("]:", 1)
                speaker = speaker.strip("[]")
                if speaker not in speakers:
                    speakers[speaker] = []
                speakers[speaker].append(content.strip())

    summaries = {}
    for speaker, texts in speakers.items():
        combined_text = " ".join(texts)
        summaries[speaker] = summarize_text(combined_text)
    return summaries

def save_summary(summaries, output_path="final_summary.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        for speaker, summary in summaries.items():
            f.write(f"{speaker} Summary:\n{summary}\n\n")
    print(f"Final summary saved at {output_path}")

if __name__ == "__main__":
    diarized_file = "diarized_transcript.txt"
    transcript_text = load_diarized_transcript(diarized_file)
    summaries = speaker_wise_summary(transcript_text)
    save_summary(summaries)