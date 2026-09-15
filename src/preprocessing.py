import cv2
import numpy as np


IMG_SIZE = (224, 224)


def preprocess_image(image_path):
    """
    Loads and preprocesses a retinal fundus image.

    Returns:
        original: RGB image resized to 224x224
        processed: normalized image suitable for EfficientNet
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # OpenCV loads BGR, convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(image, IMG_SIZE)

    # Keep copy for visualization
    original = image.copy()

    # Normalize to 0-1
    processed = image.astype(np.float32) / 255.0

    return original, processed


def load_and_preprocess(image_path):
    """
    Returns image in model-ready shape:
    (1, 224, 224, 3)
    """

    _, image = preprocess_image(image_path)

    return np.expand_dims(image, axis=0)