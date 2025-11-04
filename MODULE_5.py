import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def load_file(file_path):
    if not os.path.exists(file_path):
        return f"{file_path} not found.\n"
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def save_txt_report(audio_file, diarized_text, summary_text, output_txt="final_report.txt"):
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=== Final Speech Analysis Report ===\n\n")
        f.write(f"Audio File: {audio_file}\n\n")
        f.write("----- Speaker Diarization Output -----\n\n")
        f.write(diarized_text + "\n\n")
        f.write("----- Summarized Report -----\n\n")
        f.write(summary_text + "\n\n")
    print(f"Text report saved at {output_txt}")

def save_pdf_report(audio_file, diarized_text, summary_text, output_pdf="final_report.pdf"):
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 80, "Final Speech Analysis Report")

    c.setFont("Helvetica", 12)
    text_object = c.beginText(50, height - 120)
    text_object.textLines([
        f"Audio File: {audio_file}",
        "",
        "----- Speaker Diarization Output -----",
        diarized_text,
        "",
        "----- Summarized Report -----",
        summary_text
    ])
    c.drawText(text_object)
    c.save()
    print(f"PDF report saved at {output_pdf}")

if __name__ == "__main__":
    audio_file = "input_audio.wav"
    diarized_text = load_file("diarized_transcript.txt")
    summary_text = load_file("final_summary.txt")

    save_txt_report(audio_file, diarized_text, summary_text)
    save_pdf_report(audio_file, diarized_text, summary_text)
