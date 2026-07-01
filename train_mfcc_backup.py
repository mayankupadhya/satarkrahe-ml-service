import os
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_DIR = "final_dataset_v5/train"

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    return np.hstack([mfcc_mean, mfcc_std])

X = []
y = []

for label_name, label_value in [("fake", 0), ("real", 1)]:
    folder = os.path.join(DATA_DIR, label_name)

    for file in os.listdir(folder):
        if file.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
            path = os.path.join(folder, file)
            try:
                features = extract_mfcc(path)
                X.append(features)
                y.append(label_value)
                print("Processed:", file)
            except Exception as e:
                print("Error:", file, e)

X = np.array(X)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

pred = model.predict(X_val)
acc = accuracy_score(y_val, pred)

print("MFCC Validation Accuracy:", round(acc * 100, 2), "%")

joblib.dump(model, "mfcc_model.pkl")
print("✅ MFCC model saved as mfcc_model.pkl")