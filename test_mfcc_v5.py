import os
import joblib
import librosa
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_PATH = "mfcc_model_v5.pkl"
TEST_DIR = "final_dataset/test"  # yaha apna test folder path daal

model = joblib.load(MODEL_PATH)

def load_audio_fixed(file_path, duration=10):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    target_length = sr * duration
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    return y, sr

def mean_std(feature):
    return np.hstack([np.mean(feature, axis=1), np.std(feature, axis=1)])

def extract_features(file_path):
    y, sr = load_audio_fixed(file_path)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    return np.hstack([
        mean_std(mfcc),
        mean_std(chroma),
        mean_std(contrast),
        mean_std(centroid),
        mean_std(bandwidth),
        mean_std(rolloff),
        mean_std(zcr),
        mean_std(rms),
    ])

X_test = []
y_test = []

for label_name, label_value in [("fake", 0), ("real", 1)]:
    folder = os.path.join(TEST_DIR, label_name)

    for file in os.listdir(folder):
        if file.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm")):
            path = os.path.join(folder, file)
            try:
                X_test.append(extract_features(path))
                y_test.append(label_value)
                print("Tested:", file)
            except Exception as e:
                print("Error:", file, e)

X_test = np.array(X_test)
y_test = np.array(y_test)

pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=["FAKE", "REAL"]))