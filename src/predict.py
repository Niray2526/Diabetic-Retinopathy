import sys
import numpy as np
import tensorflow as tf

from preprocessing import load_and_preprocess


MODEL_PATH = "models/best_model.keras"

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]


def predict_image(image_path):

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    image = load_and_preprocess(
        image_path
    )

    predictions = model.predict(
        image,
        verbose=0
    )[0]

    predicted_class = np.argmax(
        predictions
    )

    confidence = predictions[
        predicted_class
    ] * 100

    print("\nPrediction")
    print("--------------------------")

    print(
        f"Class: {CLASS_NAMES[predicted_class]}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print("\nClass probabilities:")

    for name, probability in zip(
        CLASS_NAMES,
        predictions
    ):
        print(
            f"{name}: {probability * 100:.2f}%"
        )


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python predict.py image_path"
        )

        sys.exit(1)

    predict_image(
        sys.argv[1]
    )