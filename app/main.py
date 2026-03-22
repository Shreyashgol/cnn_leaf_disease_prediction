import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model" / "plant_disease_prediction_model.h5"
CLASS_MAP_PATH = BASE_DIR / "class_indices.json"
CACHE_MODEL_PATH = (
    Path(tempfile.gettempdir()) / "cnn_leaf_classifier" / MODEL_PATH.name
)
IMAGE_SIZE = (224, 224)


st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
)


@st.cache_resource
def load_model():
    model_path = resolve_model_path()
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_class_names():
    with CLASS_MAP_PATH.open() as file:
        return json.load(file)


def get_remote_model_url() -> str | None:
    secret_url = st.secrets.get("MODEL_URL")
    if secret_url:
        return secret_url

    env_url = os.getenv("MODEL_URL")
    if env_url:
        return env_url

    return None


@st.cache_resource
def resolve_model_path() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH

    remote_url = get_remote_model_url()
    if not remote_url:
        raise FileNotFoundError(
            "Model file not found locally and MODEL_URL is not configured."
        )

    CACHE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_MODEL_PATH.exists():
        with st.spinner("Downloading model file for the first app startup..."):
            urlretrieve(remote_url, CACHE_MODEL_PATH)

    return CACHE_MODEL_PATH


def format_label(raw_label: str) -> str:
    return raw_label.replace("___", " - ").replace("_", " ")


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict(image: Image.Image):
    model = load_model()
    class_names = load_class_names()

    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[str(predicted_index)]
    confidence = float(predictions[predicted_index])

    confidence_table = [
        {
            "Class": format_label(class_names[str(index)]),
            "Confidence": round(float(score) * 100, 2),
        }
        for index, score in sorted(
            enumerate(predictions), key=lambda item: item[1], reverse=True
        )[:5]
    ]

    return predicted_class, confidence, confidence_table


st.title("Plant Disease Classifier")
st.write(
    "Upload a leaf image and the CNN model will predict the plant disease class."
)

with st.expander("Model details"):
    resolved_model_path = resolve_model_path()
    st.write(f"Model file: `{resolved_model_path.name}`")
    st.write("Input image size: `224 x 224`")
    st.write("Output classes: `38`")

uploaded_file = st.file_uploader(
    "Choose a leaf image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running prediction..."):
        predicted_class, confidence, confidence_table = predict(image)

    st.subheader("Prediction")
    st.success(
        f"{format_label(predicted_class)} ({confidence * 100:.2f}% confidence)"
    )

    st.subheader("Top predictions")
    st.dataframe(confidence_table, use_container_width=True, hide_index=True)
else:
    st.info("Upload a leaf image to get started.")
