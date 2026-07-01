from fastapi import FastAPI, UploadFile, File
import librosa
import numpy as np
import os
import uuid
import joblib

app = FastAPI()

model = joblib.load("mfcc_model.pkl")


def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.hstack([mfcc_mean, mfcc_std])

    return features.reshape(1, -1)


@app.get("/")
def home():
    return {
        "status": "ML Service Running with MFCC Random Forest",
        "pipeline": "Audio -> MFCC -> Random Forest -> Prediction"
    }


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)

    file_ext = audio.filename.split(".")[-1]
    unique_name = str(uuid.uuid4())

    audio_path = f"uploads/{unique_name}.{file_ext}"

    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    features = extract_mfcc(audio_path)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    fake_prob = probabilities[0]
    real_prob = probabilities[1]

    print("================================")
    print("File:", audio.filename)
    print("Prediction:", prediction)
    print("Fake Probability:", fake_prob)
    print("Real Probability:", real_prob)
    print("================================")

    if prediction == 1:
        confidence = real_prob * 100
        risk_score = fake_prob * 100

        return {
            "label": "REAL",
            "risk": "LOW",
            "score": int(risk_score),
            "confidence": round(confidence, 2),
            "message": "Voice appears genuine",
            "recommendedAction": "Allow verification process",
            "detectionMethod": "MFCC + Random Forest Analysis"
        }

    confidence = fake_prob * 100

    return {
        "label": "FAKE",
        "risk": "HIGH",
        "score": int(confidence),
        "confidence": round(confidence, 2),
        "message": "Possible AI-generated / cloned voice detected",
        "recommendedAction": "Block sensitive transaction and alert agent",
        "detectionMethod": "MFCC + Random Forest Analysis"
    }