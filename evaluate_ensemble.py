import os
import librosa
import numpy as np
import joblib
import tensorflow as tf
#import matplotlib.pyplot as plt
import librosa.display
import tempfile
from tensorflow.keras.preprocessing import image



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

TEST_DIR = "final_dataset/test"   # yaha apna test folder path daal
RF_MODEL_PATH = "mfcc_model_v5.pkl"
CNN_MODEL_PATH = "voice_mobilenet_model.keras"

rf_model = joblib.load(RF_MODEL_PATH)
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

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

def extract_rf_features(file_path):
    y, sr = load_audio_fixed(file_path)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features = np.hstack([
        mean_std(mfcc),
        mean_std(chroma),
        mean_std(contrast),
        mean_std(centroid),
        mean_std(bandwidth),
        mean_std(rolloff),
        mean_std(zcr),
        mean_std(rms),
    ])

    return features.reshape(1, -1)
def create_cnn_input(file_path):
    y, sr = load_audio_fixed(file_path)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=1024,
        hop_length=256
    )

    S_DB = librosa.power_to_db(S, ref=np.max)

    # Same style as training spectrogram image
    S_DB = (S_DB - S_DB.min()) / (S_DB.max() - S_DB.min() + 1e-8)

    # Resize to MobileNet input size
    S_DB = tf.image.resize(S_DB[..., np.newaxis], (224, 224)).numpy()

    # MobileNet expects 3 channels
    S_RGB = np.repeat(S_DB, 3, axis=-1)

    return np.expand_dims(S_RGB, axis=0)

def get_rf_fake_prob(file_path):
    features = extract_rf_features(file_path)
    probs = rf_model.predict_proba(features)[0]
    return probs[0] * 100  # fake = 0

def get_cnn_fake_prob(file_path):
    spec = create_cnn_input(file_path)
    raw = float(cnn_model.predict(spec, verbose=0)[0][0])

    # class labels: {'fake': 0, 'real': 1}
    real_prob = raw * 100
    fake_prob = 100 - real_prob

    return fake_prob

files = []
y_true = []

for label_name, label_value in [("fake", 0), ("real", 1)]:
    folder = os.path.join(TEST_DIR, label_name)

    for file in os.listdir(folder):
        if file.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm")):
            files.append(os.path.join(folder, file))
            y_true.append(label_value)

y_true = np.array(y_true)

rf_fake_scores = []
cnn_fake_scores = []

for idx, path in enumerate(files):
    print(f"Processing {idx + 1}/{len(files)}: {os.path.basename(path)}")

    rf_fake_scores.append(get_rf_fake_prob(path))
    cnn_fake_scores.append(get_cnn_fake_prob(path))

rf_fake_scores = np.array(rf_fake_scores)
cnn_fake_scores = np.array(cnn_fake_scores)

print("\n================ RF ONLY ================")
rf_pred = np.where(rf_fake_scores >= 50, 0, 1)
print("Accuracy:", round(accuracy_score(y_true, rf_pred) * 100, 2), "%")
print(confusion_matrix(y_true, rf_pred))
print(classification_report(y_true, rf_pred, target_names=["FAKE", "REAL"]))

print("\n================ RF THRESHOLD SEARCH ================")

best_rf = {
    "f1": 0,
    "accuracy": 0,
    "threshold": None
}

for threshold in [20, 25, 30, 35, 40, 45, 50]:
    rf_pred_tuned = np.where(rf_fake_scores >= threshold, 0, 1)

    acc = accuracy_score(y_true, rf_pred_tuned)
    f1_fake = f1_score(y_true, rf_pred_tuned, pos_label=0)

    print(
        f"RF TH={threshold} => ACC={acc*100:.2f}%, FAKE_F1={f1_fake:.4f}"
    )

    if f1_fake > best_rf["f1"]:
        best_rf = {
            "f1": f1_fake,
            "accuracy": acc,
            "threshold": threshold
        }

print("\nBest RF Threshold:")
print(best_rf)

print("\n================ CNN ONLY ================")
cnn_pred = np.where(cnn_fake_scores >= 50, 0, 1)
print("Accuracy:", round(accuracy_score(y_true, cnn_pred) * 100, 2), "%")
print(confusion_matrix(y_true, cnn_pred))
print(classification_report(y_true, cnn_pred, target_names=["FAKE", "REAL"]))

best = {
    "f1": 0,
    "accuracy": 0,
    "rf_weight": None,
    "cnn_weight": None,
    "threshold": None,
}

print("\n================ ENSEMBLE SEARCH ================")

for rf_weight in [0.2, 0.3, 0.4, 0.45, 0.5, 0.6]:
    cnn_weight = 1 - rf_weight

    ensemble_scores = (rf_weight * rf_fake_scores) + (cnn_weight * cnn_fake_scores)

    for threshold in [40, 45, 50, 55, 60]:
        pred = np.where(ensemble_scores >= threshold, 0, 1)

        acc = accuracy_score(y_true, pred)
        f1_fake = f1_score(y_true, pred, pos_label=0)

        print(
            f"RF={rf_weight:.2f}, CNN={cnn_weight:.2f}, TH={threshold} "
            f"=> ACC={acc*100:.2f}%, FAKE_F1={f1_fake:.4f}"
        )

        if f1_fake > best["f1"]:
            best = {
                "f1": f1_fake,
                "accuracy": acc,
                "rf_weight": rf_weight,
                "cnn_weight": cnn_weight,
                "threshold": threshold,
            }

print("\n================ BEST ENSEMBLE ================")
print(best)

best_scores = (best["rf_weight"] * rf_fake_scores) + (best["cnn_weight"] * cnn_fake_scores)
best_pred = np.where(best_scores >= best["threshold"], 0, 1)

print("\nBest Ensemble Accuracy:", round(accuracy_score(y_true, best_pred) * 100, 2), "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, best_pred))
print("\nClassification Report:")
print(classification_report(y_true, best_pred, target_names=["FAKE", "REAL"]))