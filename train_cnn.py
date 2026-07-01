import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json

DATA_DIR = "spectrograms"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
MODEL_NAME = "voice_cnn_model.h5"

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=42
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("Class labels:", train_data.class_indices)

with open("cnn_class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

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

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.35),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.45),

    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
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
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=callbacks
)

model.save(MODEL_NAME)

val_data.reset()
pred_probs = model.predict(val_data)
preds = (pred_probs > 0.5).astype(int).reshape(-1)

print("\nConfusion Matrix:")
print(confusion_matrix(val_data.classes, preds))

print("\nClassification Report:")
print(
    classification_report(
        val_data.classes,
        preds,
        target_names=["FAKE", "REAL"]
    )
)

print(f"✅ CNN model trained and saved as {MODEL_NAME}")
print("✅ Class labels saved as cnn_class_indices.json")