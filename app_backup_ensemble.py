from fastapi import FastAPI, UploadFile, File
import librosa
import numpy as np
import os
import uuid
import joblib
import tensorflow as tf

app = FastAPI(title="SatarkRahe.ai Hybrid ML Service")

# ===============================
# Load models
# ===============================
rf_model = joblib.load("mfcc_model.pkl")
cnn_model = tf.keras.models.load_model("voice_cnn_model.h5")


# ===============================
# Utility: load fixed 10-sec audio
# ===============================
def load_audio_fixed(file_path, duration=10):
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    target_length = sr * duration

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    return y, sr


# ===============================
# Model 1: MFCC Random Forest
# ===============================
def extract_mfcc_features(file_path):
    y, sr = load_audio_fixed(file_path)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.hstack([mfcc_mean, mfcc_std])

    return features.reshape(1, -1)


def predict_rf(file_path):
    features = extract_mfcc_features(file_path)

    prediction = rf_model.predict(features)[0]
    probabilities = rf_model.predict_proba(features)[0]

    fake_prob = float(probabilities[0] * 100)
    real_prob = float(probabilities[1] * 100)

    return {
        "prediction": int(prediction),
        "fake_prob": fake_prob,
        "real_prob": real_prob,
    }


# ===============================
# Model 2: Mel Spectrogram CNN
# ===============================
def create_mel_spectrogram_array(file_path):
    y, sr = load_audio_fixed(file_path)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=1024,
        hop_length=256
    )

    S_DB = librosa.power_to_db(S, ref=np.max)

    # Normalize to 0-1
    S_DB = (S_DB - S_DB.min()) / (S_DB.max() - S_DB.min() + 1e-8)

    # Resize to 128x128
    S_DB = tf.image.resize(S_DB[..., np.newaxis], (128, 128)).numpy()

    # Convert 1 channel to 3 channels because CNN was trained on RGB images
    S_RGB = np.repeat(S_DB, 3, axis=-1)

    # Shape: (1, 128, 128, 3)
    return np.expand_dims(S_RGB, axis=0)


def predict_cnn(file_path):
    spec = create_mel_spectrogram_array(file_path)

    raw_output = float(cnn_model.predict(spec, verbose=0)[0][0])

    # IMPORTANT:
    # train_data.class_indices was {'fake': 0, 'real': 1}
    # Dense sigmoid output near 1 means REAL, near 0 means FAKE
    real_prob = raw_output * 100
    fake_prob = (1 - raw_output) * 100

    return {
        "raw_output": raw_output,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
    }


# ===============================
# Ensemble Decision Engine
# ===============================
def ensemble_decision(rf_fake, cnn_fake):
    # CNN gets slightly higher weight because problem statement is spectrogram-based
    rf_weight = 0.60
    cnn_weight = 0.40

    ensemble_fake_score = (rf_weight * rf_fake) + (cnn_weight * cnn_fake)

    # Banking use-case: lower threshold to reduce fake misses
    threshold = 45

    if ensemble_fake_score >= threshold:
        return {
            "label": "FAKE",
            "risk": "HIGH",
            "score": int(round(ensemble_fake_score)),
            "confidence": round(ensemble_fake_score, 2),
            "message": "Possible AI-generated / cloned voice detected",
            "recommendedAction": "Block sensitive transaction and alert agent",
        }

    real_confidence = 100 - ensemble_fake_score

    return {
        "label": "REAL",
        "risk": "LOW",
        "score": int(round(ensemble_fake_score)),
        "confidence": round(real_confidence, 2),
        "message": "Voice appears genuine",
        "recommendedAction": "Allow verification process",
    }


@app.get("/")
def home():
    return {
        "status": "ML Service Running",
        "pipeline": "Audio -> MFCC Random Forest + Mel Spectrogram CNN -> Weighted Ensemble",
        "models": {
            "model_1": "MFCC Random Forest",
            "model_2": "Mel Spectrogram CNN",
            "ensemble": "0.45 RF + 0.55 CNN"
        }
    }


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)

    file_ext = audio.filename.split(".")[-1]
    unique_name = str(uuid.uuid4())
    audio_path = f"uploads/{unique_name}.{file_ext}"

    try:
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        rf_result = predict_rf(audio_path)
        cnn_result = predict_cnn(audio_path)

        rf_fake = rf_result["fake_prob"]
        cnn_fake = cnn_result["fake_prob"]

        final_result = ensemble_decision(rf_fake, cnn_fake)

        print("================================")
        print("File:", audio.filename)
        print("RF Fake Probability:", round(rf_fake, 2))
        print("RF Real Probability:", round(rf_result["real_prob"], 2))
        print("CNN Fake Probability:", round(cnn_fake, 2))
        print("CNN Real Probability:", round(cnn_result["real_prob"], 2))
        print("Ensemble Label:", final_result["label"])
        print("Ensemble Score:", final_result["score"])
        print("================================")

        return {
            "label": final_result["label"],
            "risk": final_result["risk"],
            "score": final_result["score"],
            "confidence": final_result["confidence"],
            "message": final_result["message"],
            "recommendedAction": final_result["recommendedAction"],
            "detectionMethod": "Hybrid Ensemble: MFCC Random Forest + Mel Spectrogram CNN",
            "rfFakeScore": round(rf_fake, 2),
            "cnnFakeScore": round(cnn_fake, 2),
            "ensembleScore": final_result["score"],
        }

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)