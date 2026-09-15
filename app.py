import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image


# =========================
# CONFIGURATION
# =========================

MODEL_PATH = "models/best_model.keras"

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]

IMAGE_SIZE = (224, 224)


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# =========================
# PREPROCESS
# =========================

def preprocess_image(image):

    image = np.array(image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    image = image.astype(
        np.float32
    ) / 255.0

    return np.expand_dims(
        image,
        axis=0
    )


# =========================
# PAGE
# =========================

st.set_page_config(
    page_title="DR-XAI",
    page_icon="🩺",
    layout="wide"
)


st.title(
    "🩺 Explainable Diabetic Retinopathy Detection"
)

st.write(
    "Upload a retinal fundus image to obtain an "
    "AI-based severity prediction."
)


# =========================
# UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload retinal image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True
        )


    if st.button(
        "🔍 Analyze Image"
    ):

        with st.spinner(
            "Analyzing retinal image..."
        ):

            model = load_model()

            processed = preprocess_image(
                image
            )

            predictions = model.predict(
                processed,
                verbose=0
            )[0]

            predicted_class = np.argmax(
                predictions
            )

            confidence = (
                predictions[predicted_class]
                * 100
            )


        with col2:

            st.subheader(
                "AI Prediction"
            )

            st.success(
                CLASS_NAMES[predicted_class]
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


        st.subheader(
            "Class Probabilities"
        )

        for name, probability in zip(
            CLASS_NAMES,
            predictions
        ):

            st.write(
                f"{name}: "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )


        st.warning(
            "This application is for educational "
            "and research purposes only. It is not "
            "a substitute for professional medical diagnosis."
        )