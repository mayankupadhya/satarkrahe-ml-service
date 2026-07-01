import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import json

DATA_DIR = "spectrograms"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
MODEL_NAME = "voice_mobilenet_model.keras"

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

with open("mobilenet_class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
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
        patience=4,
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
    epochs=15,
    callbacks=callbacks
)

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=8,
    callbacks=callbacks
)

model.save(MODEL_NAME)

val_data.reset()
pred_probs = model.predict(val_data)
preds = (pred_probs > 0.5).astype(int).reshape(-1)

print("\nValidation Accuracy:", round(accuracy_score(val_data.classes, preds) * 100, 2), "%")

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

print(f"✅ MobileNet spectrogram model saved as {MODEL_NAME}")