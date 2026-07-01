import os
import librosa
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_splittrain_cnn.py
from sklearn.metrics import accuracy_score


DATASET_DIR = "dataset"


def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(audio)

    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_centroid, axis=1),
        np.mean(zero_crossing, axis=1),
    ])

    return features


X = []
y = []

for label_name, label_value in [("real", 0), ("fake", 1)]:
    folder = os.path.join(DATASET_DIR, label_name)

    for file_name in os.listdir(folder):
        if file_name.lower().endswith((".wav", ".mp3", ".m4a", ".ogg")):
            file_path = os.path.join(folder, file_name)

            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(label_value)
                print("Processed:", file_path)
            except Exception as e:
                print("Error:", file_path, e)


X = np.array(X)
y = np.array(y)

if len(X) < 4:
    raise Exception("Not enough audio files. Add more real and fake samples.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")