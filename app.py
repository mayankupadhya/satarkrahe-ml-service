from fastapi import FastAPI, UploadFile, File
import librosa
import librosa.display
import numpy as np
import os
import uuid
import joblib
import subprocess
import io
import base64

# NEW: matplotlib for spectrogram generation (server-safe, no GUI)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = FastAPI(title="SatarkRahe.ai ML Service")

rf_model = joblib.load("mfcc_model_v6.pkl")

RF_THRESHOLD = 23


def convert_to_wav(input_path: str) -> str:
    output_path = input_path.rsplit(".", 1)[0] + "_converted.wav"
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-f", "wav", output_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
    return output_path


def load_audio_fixed(file_path, duration=10):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    actual_duration = len(y) / sr
    if actual_duration < 1.0:
        raise ValueError(f"Audio too short: {actual_duration:.2f}s — minimum 1 second required")
    use_duration = min(actual_duration, duration)
    target_length = int(sr * use_duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    return y, sr


def mean_std(feature):
    return np.hstack([np.mean(feature, axis=1), np.std(feature, axis=1)])


def generate_spectrogram_image(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.set_title('')
    ax.axis('off')
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


def extract_features_fast(y_window, sr):
    """
    Lightweight feature extraction for window-level artifact scanning.
    Pyin skip kiya hai (bohot slow) — sirf location dhoondhne ke liye.
    f0_mean_std ke liye zeros daale hain taaki feature shape (292) match rahe.
    """
    mfcc        = librosa.feature.mfcc(y=y_window, sr=sr, n_mfcc=40)
    chroma      = librosa.feature.chroma_stft(y=y_window, sr=sr)
    contrast    = librosa.feature.spectral_contrast(y=y_window, sr=sr)
    centroid    = librosa.feature.spectral_centroid(y=y_window, sr=sr)
    bandwidth   = librosa.feature.spectral_bandwidth(y=y_window, sr=sr)
    rolloff     = librosa.feature.spectral_rolloff(y=y_window, sr=sr)
    zcr         = librosa.feature.zero_crossing_rate(y_window)
    rms         = librosa.feature.rms(y=y_window)
    mfcc_delta  = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    flatness    = librosa.feature.spectral_flatness(y=y_window)
    f0_mean_std = np.array([0.0, 0.0])

    features = np.hstack([
        mean_std(mfcc), mean_std(chroma), mean_std(contrast),
        mean_std(centroid), mean_std(bandwidth), mean_std(rolloff),
        mean_std(zcr), mean_std(rms), mean_std(mfcc_delta),
        mean_std(mfcc_delta2), mean_std(flatness), f0_mean_std,
    ])
    return features.reshape(1, -1)


def find_artifact_location(y, sr, window_sec=1.5):
    """
    Audio ko overlapping 1.5-sec windows mein todta hai, har window pe
    fast prediction karta hai, aur highest fake-confidence wala window
    ka timestamp return karta hai — ye REAL artifact location hai.
    """
    window_size = int(sr * window_sec)
    hop_size    = int(sr * 0.5)

    best_time       = 0.0
    best_fake_score = 0.0
    window_scores   = []

    start = 0
    while start + window_size <= len(y):
        y_win     = y[start: start + window_size]
        t_sec     = round(start / sr, 2)

        try:
            feats      = extract_features_fast(y_win, sr)
            probs      = rf_model.predict_proba(feats)[0]
            fake_score = float(probs[0] * 100)
        except Exception:
            fake_score = 0.0

        window_scores.append({"time": t_sec, "fakeScore": round(fake_score, 2)})

        if fake_score > best_fake_score:
            best_fake_score = fake_score
            best_time       = t_sec

        start += hop_size

    return {
        "artifactTime":       round(best_time, 2),
        "artifactConfidence": round(best_fake_score, 2),
        "windowScores":       window_scores,
    }


def extract_features(file_path):
    y, sr = load_audio_fixed(file_path)

    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast    = librosa.feature.spectral_contrast(y=y, sr=sr)
    centroid    = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth   = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff     = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr         = librosa.feature.zero_crossing_rate(y)
    rms         = librosa.feature.rms(y=y)
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    flatness    = librosa.feature.spectral_flatness(y=y)

    f0, _, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'), sr=sr
    )
    voiced_f0 = f0[~np.isnan(f0)]
    f0_mean_std = np.array([np.mean(voiced_f0), np.std(voiced_f0)]) if voiced_f0.size > 0 else np.array([0.0, 0.0])

    features = np.hstack([
        mean_std(mfcc), mean_std(chroma), mean_std(contrast),
        mean_std(centroid), mean_std(bandwidth), mean_std(rolloff),
        mean_std(zcr), mean_std(rms), mean_std(mfcc_delta),
        mean_std(mfcc_delta2), mean_std(flatness), f0_mean_std,
    ])

    spectrogram_image = generate_spectrogram_image(y, sr)
    artifact_info     = find_artifact_location(y, sr)

    return features.reshape(1, -1), spectrogram_image, artifact_info, y, sr


@app.get("/")
def home():
    return {
        "status": "ML Service Running",
        "pipeline": "Audio -> MFCC Acoustic Features -> Random Forest -> Tuned Banking Risk Decision",
        "model": "MFCC Acoustic Random Forest v6",
        "threshold": RF_THRESHOLD
    }


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)

    file_ext     = audio.filename.split(".")[-1].lower()
    unique_name  = str(uuid.uuid4())
    audio_path   = f"uploads/{unique_name}.{file_ext}"
    converted_path = None

    try:
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        if file_ext != "wav":
            print(f"Converting {file_ext} -> WAV using FFmpeg...")
            converted_path = convert_to_wav(audio_path)
            process_path   = converted_path
        else:
            process_path = audio_path

        features, spectrogram_image, artifact_info, y, sr = extract_features(process_path)

        # NEW: actual audio duration in seconds (real length, not the padded 10-sec window)
        duration_sec = round(len(y) / sr, 2)

        probs     = rf_model.predict_proba(features)[0]
        fake_prob = float(probs[0] * 100)
        real_prob = float(probs[1] * 100)

        if fake_prob >= RF_THRESHOLD:
            label              = "FAKE"
            risk               = "HIGH"
            score              = int(round(fake_prob))
            confidence         = round(fake_prob, 2)
            message            = "Possible AI-generated / cloned voice detected"
            recommended_action = "Block sensitive transaction and alert agent"
        else:
            label              = "REAL"
            risk               = "LOW"
            score              = int(round(fake_prob))
            confidence         = round(real_prob, 2)
            message            = "Voice appears genuine"
            recommended_action = "Allow verification process"

        print("================================")
        print(f"File     : {audio.filename}")
        print(f"Format   : {file_ext}")
        print(f"Duration : {duration_sec}s")
        print(f"Fake %   : {round(fake_prob, 2)}")
        print(f"Real %   : {round(real_prob, 2)}")
        print(f"Threshold: {RF_THRESHOLD}")
        print(f"Label    : {label}")
        print(f"Artifact : {artifact_info['artifactTime']}s @ {artifact_info['artifactConfidence']}% fake")
        print("================================")

        return {
            "label":             label,
            "risk":              risk,
            "score":             score,
            "confidence":        confidence,
            "message":           message,
            "recommendedAction": recommended_action,
            "detectionMethod":   "MFCC Acoustic Random Forest v6 + FFmpeg Normalized Input",
            "fakeProbability":   round(fake_prob, 2),
            "realProbability":   round(real_prob, 2),
            "threshold":         RF_THRESHOLD,
            "spectrogramImage":  spectrogram_image,
            "artifactTime":      artifact_info["artifactTime"],
            "artifactConfidence":artifact_info["artifactConfidence"],
            "windowScores":      artifact_info["windowScores"],
            "durationSec":       duration_sec,
        }

    except ValueError as ve:
        return {
            "label": "UNKNOWN", "risk": "LOW", "score": 0, "confidence": 0,
            "message": str(ve), "recommendedAction": "Please speak for at least 1 second",
            "detectionMethod": "N/A", "fakeProbability": 0, "realProbability": 0,
            "threshold": RF_THRESHOLD, "spectrogramImage": None,
            "artifactTime": None, "artifactConfidence": None, "windowScores": [],
            "durationSec": None,
        }

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)