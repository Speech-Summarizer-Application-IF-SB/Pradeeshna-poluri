import queue
import sounddevice as sd
import vosk
import sys
import json
from jiwer import wer

model = vosk.Model("c:/users/SYS/desktop/vosk-model-small-en-us-0.15")
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

device_info = sd.query_devices(kind='input')
samplerate = 16000
rec = vosk.KaldiRecognizer(model, samplerate)

print("Start speaking... (Press Ctrl+c to stop)\n")

transcriptions = []

try:
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=2, callback=callback, device=1):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    print("Recognized:", text)
                    transcriptions.append(text)
            else:
                partial = json.loads(rec.PartialResult())
                if partial.get("partial"):
                    print("... ", partial["partial"], end="\r")

except KeyboardInterrupt:
    print("\nStopped recording.")
    final_result = json.loads(rec.FinalResult())
    if final_result.get("text"):
        transcriptions.append(final_result["text"])

reference_text = "hello world"
hypothesis_text = " ".join(transcriptions)
error_rate = wer(reference_text, hypothesis_text)

print("\nEvaluation Report")
print("Reference :", reference_text)
print("Hypothesis:", hypothesis_text)
print(f"WER       : {error_rate*100:.2f}%")