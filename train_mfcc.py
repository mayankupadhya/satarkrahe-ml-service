import os
import re
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "final_dataset_v5/train"
MODEL_NAME = "mfcc_model_v6.pkl"
TARGET_SR = 16000  # adjust if your load_audio_fixed uses a different rate


# ============================================================
# AUDIO LOADING
# ============================================================
def load_audio_fixed(file_path, sr=TARGET_SR):
    """
    Load audio at a fixed sample rate.
    NOTE: replace this with your existing load_audio_fixed if you already
    have one elsewhere (e.g. duration padding/truncation logic from app.py).
    Keeping this consistent between train_mfcc.py and app.py is critical.
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr


# ============================================================
# HELPERS
# ============================================================
def mean_std(feature_matrix):
    """Collapse a (n_features, n_frames) matrix into mean+std per row."""
    return np.hstack([
        np.mean(feature_matrix, axis=1),
        np.std(feature_matrix, axis=1)
    ])


def safe_pyin_f0(y, sr):
    """
    Extract F0 contour with pyin and return mean/std of the VOICED frames only.
    pyin returns NaN for unvoiced frames - these must be filtered out before
    aggregating, otherwise a single NaN poisons the mean/std for the whole file.
    """
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
    except Exception:
        # If pyin fails outright (e.g. very short/silent clip), return zeros
        return np.array([0.0, 0.0])

    voiced_f0 = f0[~np.isnan(f0)]

    if voiced_f0.size == 0:
        # No voiced frames detected at all - return zeros rather than NaN
        return np.array([0.0, 0.0])

    return np.array([np.mean(voiced_f0), np.std(voiced_f0)])


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_features(file_path):
    y, sr = load_audio_fixed(file_path)

    # --- existing v5 features ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    # --- NEW v6 features ---
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    flatness = librosa.feature.spectral_flatness(y=y)
    f0_mean_std = safe_pyin_f0(y, sr)

    features = np.hstack([
        mean_std(mfcc),
        mean_std(chroma),
        mean_std(contrast),
        mean_std(centroid),
        mean_std(bandwidth),
        mean_std(rolloff),
        mean_std(zcr),
        mean_std(rms),
        mean_std(mfcc_delta),
        mean_std(mfcc_delta2),
        mean_std(flatness),
        f0_mean_std,
    ])

    return features


# ============================================================
# SPEAKER ID EXTRACTION (for leakage-free split)
# ============================================================
def get_speaker_id(filename):
    """
    Pull an idXXXXX-style speaker token out of the filename if present.
    Falls back to the filename itself (so files with no id are treated
    as their own isolated group rather than silently grouped together).
    Examples this should catch:
      00001_id00518_qm7oQcDg4oI_id02268_wavtolip.wav -> multiple ids present;
        we take the FIRST id as the anchor speaker for grouping purposes.
      LA_T_1012129.flac -> no id pattern, falls back to full filename
      common_voice_hi_23795243.mp3 -> no id pattern, falls back to full filename
    """
    match = re.search(r'id(\d+)', filename)
    if match:
        return f"id{match.group(1)}"
    return filename  # no speaker id found - treat as its own group


# ============================================================
# SOURCE-LEAKAGE FILTER
# ============================================================
# IMPORTANT: in this dataset, ALL .flac files (ASVspoof) ended up in the
# fake/ class and ALL common_voice_hi_*.mp3 files (Common Voice Hindi)
# ended up in the real/ class. That means file-format/codec became a
# perfect proxy for the label - the model could hit 99%+ accuracy just
# by detecting FLAC-vs-MP3 encoding artifacts, never learning real
# voice-authenticity signal. We exclude both sources here so training
# only uses the wavtolip/faceswap (FakeAVCeleb-style) data, where the
# same speaker IDs and same WAV format genuinely appear in both classes.
def is_excluded_source(filename):
    lower = filename.lower()
    if lower.endswith(".flac"):
        return True
    if lower.startswith("common_voice_hi_"):
        return True
    return False


# ============================================================
# BUILD DATASET
# ============================================================
X = []
y_labels = []
speaker_ids = []
skipped_count = 0

for label_name, label_value in [("fake", 0), ("real", 1)]:
    folder = os.path.join(DATA_DIR, label_name)

    for file in os.listdir(folder):
        if file.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm")):

            if is_excluded_source(file):
                skipped_count += 1
                continue

            path = os.path.join(folder, file)

            try:
                features = extract_features(path)
                X.append(features)
                y_labels.append(label_value)
                speaker_ids.append(get_speaker_id(file))
                print("Processed:", file)
            except Exception as e:
                print("Error:", file, e)

print("\nSkipped (excluded source - flac/common_voice):", skipped_count)

X = np.array(X)
y_labels = np.array(y_labels)
speaker_ids = np.array(speaker_ids)

print("Dataset shape:", X.shape)
print("Labels shape:", y_labels.shape)
print("Fake count:", np.sum(y_labels == 0))
print("Real count:", np.sum(y_labels == 1))
print("Unique speaker groups:", len(np.unique(speaker_ids)))


# ============================================================
# SPEAKER-LEVEL TRAIN/VAL SPLIT (no leakage across classes/files)
# ============================================================
unique_speakers = np.unique(speaker_ids)

train_speakers, val_speakers = train_test_split(
    unique_speakers,
    test_size=0.2,
    random_state=42
)

train_mask = np.isin(speaker_ids, train_speakers)
val_mask = np.isin(speaker_ids, val_speakers)

X_train, y_train = X[train_mask], y_labels[train_mask]
X_val, y_val = X[val_mask], y_labels[val_mask]

print("\nTrain samples:", X_train.shape[0], " Val samples:", X_val.shape[0])
print("Train fake/real:", np.sum(y_train == 0), "/", np.sum(y_train == 1))
print("Val fake/real:", np.sum(y_val == 0), "/", np.sum(y_val == 1))


# ============================================================
# TRAIN MODEL
# ============================================================
model = RandomForestClassifier(
    n_estimators=700,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_val)
acc = accuracy_score(y_val, pred)

print("\n==============================")
print("Improved MFCC Validation Accuracy:", round(acc * 100, 2), "%")
print("==============================\n")

print("Confusion Matrix:")
print(confusion_matrix(y_val, pred))

print("\nClassification Report:")
print(classification_report(y_val, pred, target_names=["FAKE", "REAL"]))

joblib.dump(model, MODEL_NAME)

print(f"✅ Improved acoustic model saved as {MODEL_NAME}")