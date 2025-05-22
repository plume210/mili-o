import os
import numpy as np
from TTS.api import TTS
from pydub import AudioSegment
import soundfile as sf

# 1. Init modèle TTS multilingue
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)

# 2. Créer un répertoire de sortie
os.makedirs("synthetic_corpus/train/audio", exist_ok=True)
os.makedirs("synthetic_corpus/train/rttm", exist_ok=True)

# 3. Génère deux voix (FR et EN)
speakers = {
    "spk1": {"text": "Bonjour, tu vas bien ?", "lang": "fr"},
    "spk2": {"text": "Yes, I'm doing great, thank you!", "lang": "en"},
    "spk3": {"text": "Qu'est-ce que tu fais ce soir ?", "lang": "fr"},
    "spk4": {"text": "Probably going to watch a movie.", "lang": "en"},
}

segments = []
combined = AudioSegment.silent(duration=1000)

start_time = 1.0
for i, (spk, info) in enumerate(speakers.items()):
    wav = tts.tts(info["text"], speaker=tts.speakers[0], language=info["lang"], return_type="np")
    duration = len(wav) / tts.sample_rate
    filename = f"temp_{spk}.wav"
    sf.write(filename, wav, tts.sample_rate)
    audio = AudioSegment.from_wav(filename)

    # RTTM compatible info
    segments.append(f"SPEAKER dialogue1 1 {start_time:.2f} {duration:.2f} <NA> <NA> {spk} <NA> <NA>")
    combined += audio + AudioSegment.silent(duration=300)
    start_time += duration + 0.3

# 4. Sauvegarder le fichier final
combined.export("synthetic_corpus/train/audio/dialogue1.wav", format="wav")

with open("synthetic_corpus/train/rttm/dialogue1.rttm", "w") as f:
    for line in segments:
        f.write(line + "\n")

print("✅ Données synthétiques prêtes !")
