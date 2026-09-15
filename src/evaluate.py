import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

DATASET_DIR = "dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "train_images")
CSV_FILE = os.path.join(DATASET_DIR, "train.csv")

MODEL_PATH = "models/best_model.keras"

os.makedirs("outputs", exist_ok=True)


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(CSV_FILE)

df["image_path"] = df["id_code"].apply(
    lambda x: os.path.join(IMAGE_DIR, f"{x}.png")
)

df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)


_, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["diagnosis"],
    random_state=42
)


# =========================
# LOAD IMAGE
# =========================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_png(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image, label


dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_df["image_path"].values,
        test_df["diagnosis"].values
    )
)

dataset = (
    dataset
    .map(load_image)
    .batch(BATCH_SIZE)
)


# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# =========================
# PREDICT
# =========================

probabilities = model.predict(dataset)

predictions = np.argmax(
    probabilities,
    axis=1
)

actual = test_df["diagnosis"].values


# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(
    actual,
    predictions
)

print(f"\nAccuracy: {accuracy:.4f}")


# =========================
# CLASSIFICATION REPORT
# =========================

class_names = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

print(
    classification_report(
        actual,
        predictions,
        target_names=class_names,
        zero_division=0
    )
)


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    actual,
    predictions
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    xticks_rotation=45
)

plt.title(
    "Diabetic Retinopathy Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.show()

print(
    "Confusion matrix saved to outputs/confusion_matrix.png"
)