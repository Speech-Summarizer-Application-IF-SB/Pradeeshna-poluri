import os, warnings, torchaudio
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_AUTO"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface_no_symlink")
from dotenv import load_dotenv
import os
load_dotenv()
import torch
import warnings
warnings.filterwarnings("ignore")
from pyannote.audio import Pipeline
import speechbrain as sb
def speaker_diarization(audio_path, output_txt="diarized_transcript.txt"):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    token = os.getenv("HUGGINGFACE_TOKEN")
    print("Token Loaded:", bool(token))  # This should print True if it loaded correctly
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
    diarization_result = pipeline(audio_path)
    transcript = ""
    
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        start_time = round(turn.start, 2)
        end_time = round(turn.end, 2)
        line = f"[{speaker}] ({start_time}s - {end_time}s)"
        transcript += f"{line}\n"

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(transcript)

    print(f"Results saved to '{output_txt}'")
    return transcript

if __name__ == "__main__":
    audio_file = r"c:\Users\SYS\Desktop\streamlit_project\speech_project\meeting_audio.wav.wav"
    try:
        speaker_diarization(audio_file)
    except Exception as e:
        print(f"Error: {e}")