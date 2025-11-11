import os
print("Looking in folder:")
os.getcwd()
print("Files here:")
os.listdir()

# stt_from_file.py
import wave
import json
import os
import vosk

model = vosk.Model("models/vosk-model-small-en-us-0.15")
wf = wave.open(r"c:\Users\SYS\Downloads\sample.wav.wav") # put a small mono 16kHz wav here

if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
    print("Please use a mono 16kHz WAV (convert with Audacity or ffmpeg).")
    # you can still try but results may be poor

rec = vosk.KaldiRecognizer(model, wf.getframerate())
results = []
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        results.append(json.loads(rec.Result()))
final = json.loads(rec.FinalResult())
results.append(final)
text = " ".join(r.get("text", "") for r in results)
print("Transcript:", text)
 