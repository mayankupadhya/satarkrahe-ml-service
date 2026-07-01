import os
import librosa
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_DIR = "final_dataset_v5/train"
MODEL_NAME = "voice_cnn_numpy_model.h5"

DURATION = 10
SR = 16000
IMG_SIZE = 128

def load_audio_fixed(file_path):
    y, sr = librosa.load(file_path, sr=SR, mono=True)

    target_length = SR * DURATION

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    return y, sr

def extract_mel_array(file_path):
    y, sr = load_audio_fixed(file_path)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=1024,
        hop_length=256
    )

    S_DB = librosa.power_to_db(S, ref=np.max)

    S_DB = (S_DB - S_DB.min()) / (S_DB.max() - S_DB.min() + 1e-8)

    S_DB = tf.image.resize(S_DB[..., np.newaxis], (IMG_SIZE, IMG_SIZE)).numpy()

    return S_DB

X = []
y = []

for label_name, label_value in [("fake", 0), ("real", 1)]:
    folder = os.path.join(DATA_DIR, label_name)

    for file in os.listdir(folder):
        if file.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm")):
            path = os.path.join(folder, file)

            try:
                features = extract_mel_array(path)
                X.append(features)
                y.append(label_value)
                print("Processed:", file)
            except Exception as e:
                print("Error:", file, e)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Fake count:", np.sum(y == 0))
print("Real count:", np.sum(y == 1))

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = models.Sequential([
    layers.Input(shape=(128, 128, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.20),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.30),

    layers.GlobalAveragePooling2D(),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.45),

    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=0.00001
    ),
    ModelCheckpoint(
        MODEL_NAME,
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    callbacks=callbacks
)

model.save(MODEL_NAME)

pred_probs = model.predict(X_val)
preds = (pred_probs > 0.5).astype(int).reshape(-1)

print("\nValidation Accuracy:", round(accuracy_score(y_val, preds) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, preds))

print("\nClassification Report:")
print(classification_report(y_val, preds, target_names=["FAKE", "REAL"]))

print(f"✅ NumPy Mel CNN saved as {MODEL_NAME}")