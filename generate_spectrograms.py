import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = "final_dataset_v5/train"
OUTPUT_DIR = "spectrograms"

os.makedirs(f"{OUTPUT_DIR}/real", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/fake", exist_ok=True)

def create_spectrogram(audio_path, output_path, duration=10):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    target_length = sr * duration

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=1024,
        hop_length=256
    )

    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_DB, sr=sr, cmap="magma")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

for label in ["real", "fake"]:
    folder = os.path.join(INPUT_DIR, label)

    for file in os.listdir(folder):
        if file.lower().endswith((".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac")):
            input_path = os.path.join(folder, file)
            output_name = os.path.splitext(file)[0] + ".png"
            output_path = os.path.join(OUTPUT_DIR, label, output_name)

            create_spectrogram(input_path, output_path)
            print(f"Created: {output_path}")

print("✅ Spectrogram generation complete")