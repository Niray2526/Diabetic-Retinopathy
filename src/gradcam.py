import os
import sys

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


MODEL_PATH = "models/best_model.keras"

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

IMAGE_SIZE = (224, 224)


def load_image(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    processed = (
        image.astype(np.float32) / 255.0
    )

    return image, np.expand_dims(
        processed,
        axis=0
    )


def find_last_conv_layer(model):

    for layer in reversed(model.layers):

        try:

            shape = layer.output.shape

            if len(shape) == 4:

                return layer.name

        except Exception:
            continue

    raise ValueError(
        "Could not find convolutional layer."
    )


def make_gradcam(model, image):

    last_conv_layer_name = (
        find_last_conv_layer(model)
    )

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(
                last_conv_layer_name
            ).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = (
            grad_model(image)
        )

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_score = predictions[
            0,
            predicted_class
        ]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return (
        heatmap.numpy(),
        predicted_class.numpy(),
        predictions[0].numpy()
    )


def save_gradcam(
    image_path,
    output_path="outputs/gradcam.png"
):

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    original, processed = load_image(
        image_path
    )

    heatmap, predicted_class, probabilities = (
        make_gradcam(
            model,
            processed
        )
    )

    # Convert heatmap to uint8
    heatmap = np.uint8(
        255 * heatmap
    )

    # Resize heatmap
    heatmap = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    # Apply OpenCV colormap
    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # Blend with original
    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    confidence = (
        probabilities[predicted_class]
        * 100
    )

    # Create figure
    plt.figure(
        figsize=(12, 5)
    )

    plt.subplot(
        1,
        2,
        1
    )

    plt.imshow(
        original
    )

    plt.title(
        "Original Image"
    )

    plt.axis("off")

    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(
        overlay
    )

    plt.title(
        f"Grad-CAM\n"
        f"{CLASS_NAMES[predicted_class]} "
        f"({confidence:.2f}%)"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Grad-CAM saved to: {output_path}"
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python gradcam.py image_path"
        )

        sys.exit(1)

    save_gradcam(
        sys.argv[1]
    )