from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np

model = load_model("voice_cnn_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0

    score = model.predict(arr, verbose=0)[0][0]
    prediction = "real" if score >= 0.5 else "fake"
    return prediction, score

stats = {
    "real": {"correct": 0, "total": 0},
    "fake": {"correct": 0, "total": 0},
}

wrong = []

for label in ["real", "fake"]:
    folder = f"test_spectrograms/{label}"

    for file in os.listdir(folder):
        if file.lower().endswith(".png"):
            pred, score = predict_image(os.path.join(folder, file))

            stats[label]["total"] += 1

            if pred == label:
                stats[label]["correct"] += 1
            else:
                wrong.append((file, label, pred, score))

total_correct = stats["real"]["correct"] + stats["fake"]["correct"]
total = stats["real"]["total"] + stats["fake"]["total"]

print("\n===== RESULT =====")
print("Overall Accuracy:", round((total_correct / total) * 100, 2), "%")
print("Correct:", total_correct)
print("Total:", total)

print("\n===== CLASS WISE =====")
for label in ["real", "fake"]:
    correct = stats[label]["correct"]
    total_label = stats[label]["total"]
    acc = round((correct / total_label) * 100, 2)
    print(f"{label.upper()} Accuracy: {acc}% ({correct}/{total_label})")

print("\n===== WRONG SAMPLES FIRST 20 =====")
for item in wrong[:20]:
    file, actual, pred, score = item
    print(f"Wrong: {file} | Actual: {actual} | Predicted: {pred} | Score: {score}")

if pred != label:
    print(
        f"Wrong: {file} | Actual: {label} | Predicted: {pred} | Score: {score:.4f}"
    )