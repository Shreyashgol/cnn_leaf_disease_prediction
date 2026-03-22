import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlretrieve

import gdown
import h5py
import numpy as np
import requests
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
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except (TypeError, ValueError) as error:
        if "quantization_config" not in str(error):
            raise
        return load_model_without_quantization_config(model_path)


@st.cache_data
def load_class_names():
    with CLASS_MAP_PATH.open() as file:
        return json.load(file)


def get_remote_model_url() -> str | None:
    secret_url = st.secrets.get("MODEL_URL")
    if secret_url:
        return normalize_model_url(secret_url)

    env_url = os.getenv("MODEL_URL")
    if env_url:
        return normalize_model_url(env_url)

    return None


def get_model_download_token() -> str | None:
    secret_token = st.secrets.get("MODEL_DOWNLOAD_TOKEN")
    if secret_token:
        return secret_token

    env_token = os.getenv("MODEL_DOWNLOAD_TOKEN")
    if env_token:
        return env_token

    return None


def normalize_model_url(url: str) -> str:
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc:
        return url

    path_parts = [part for part in parsed.path.split("/") if part]
    if "file" in path_parts and "d" in path_parts:
        file_id_index = path_parts.index("d") + 1
        if file_id_index < len(path_parts):
            file_id = path_parts[file_id_index]
            return f"https://drive.google.com/uc?export=download&id={file_id}"

    query_file_id = parse_qs(parsed.query).get("id", [None])[0]
    if query_file_id:
        return f"https://drive.google.com/uc?export=download&id={query_file_id}"

    return url


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
            download_model_file(remote_url, CACHE_MODEL_PATH)

    validate_model_file(CACHE_MODEL_PATH)

    return CACHE_MODEL_PATH


def download_model_file(remote_url: str, destination: Path) -> None:
    if "drive.google.com" in remote_url:
        gdown.download(remote_url, str(destination), quiet=False, fuzzy=True)
        return

    download_token = get_model_download_token()
    if download_token:
        download_model_file_with_token(remote_url, destination, download_token)
        return

    try:
        urlretrieve(remote_url, destination)
    except HTTPError as error:
        if error.code == 401:
            raise PermissionError(
                "Model download returned HTTP 401 Unauthorized. "
                "Use a public direct-download URL, or set MODEL_DOWNLOAD_TOKEN "
                "for a protected host such as Hugging Face."
            ) from error
        raise


def download_model_file_with_token(
    remote_url: str, destination: Path, download_token: str
) -> None:
    headers = {"Authorization": f"Bearer {download_token}"}
    with requests.get(remote_url, headers=headers, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output_file.write(chunk)


def validate_model_file(model_path: Path) -> None:
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise FileNotFoundError("Model download failed or created an empty file.")

    if not h5py.is_hdf5(model_path):
        model_path.unlink(missing_ok=True)
        raise ValueError(
            "Downloaded file is not a valid H5 model. "
            "If you are using Google Drive, make sure the file is shared as "
            "'Anyone with the link can view'."
        )


def load_model_without_quantization_config(model_path: Path):
    with h5py.File(model_path, "r") as model_file:
        model_config = model_file.attrs.get("model_config")

    if model_config is None:
        raise ValueError("Could not find model_config in the H5 model file.")

    if isinstance(model_config, bytes):
        model_config = model_config.decode("utf-8")

    cleaned_config = remove_quantization_config(json.loads(model_config))
    model = build_model_from_legacy_config(cleaned_config)
    model.load_weights(model_path)
    return model


def remove_quantization_config(config):
    cleaned = deepcopy(config)

    def walk(value):
        if isinstance(value, dict):
            value.pop("quantization_config", None)
            for nested_value in value.values():
                walk(nested_value)
        elif isinstance(value, list):
            for nested_value in value:
                walk(nested_value)

    walk(cleaned)
    return cleaned


def build_model_from_legacy_config(model_config):
    if model_config.get("class_name") != "Sequential":
        raise ValueError("Only Sequential legacy models are supported.")

    sequential_config = model_config["config"]
    model = tf.keras.Sequential(name=sequential_config.get("name", "sequential"))

    for layer in sequential_config.get("layers", []):
        class_name = layer["class_name"]
        config = layer["config"]

        if class_name == "InputLayer":
            batch_shape = config.get("batch_shape")
            if not batch_shape or len(batch_shape) < 2:
                raise ValueError("InputLayer batch_shape is missing from model config.")
            model.add(
                tf.keras.layers.InputLayer(
                    input_shape=tuple(batch_shape[1:]),
                    name=config.get("name"),
                    dtype=deserialize_dtype(config.get("dtype")),
                )
            )
        elif class_name == "Conv2D":
            model.add(
                tf.keras.layers.Conv2D(
                    filters=config["filters"],
                    kernel_size=tuple(config["kernel_size"]),
                    strides=tuple(config.get("strides", [1, 1])),
                    padding=config.get("padding", "valid"),
                    data_format=config.get("data_format", "channels_last"),
                    dilation_rate=tuple(config.get("dilation_rate", [1, 1])),
                    groups=config.get("groups", 1),
                    activation=config.get("activation"),
                    use_bias=config.get("use_bias", True),
                    kernel_initializer=deserialize_initializer(
                        config.get("kernel_initializer")
                    ),
                    bias_initializer=deserialize_initializer(
                        config.get("bias_initializer")
                    ),
                    name=config.get("name"),
                    dtype=deserialize_dtype(config.get("dtype")),
                    trainable=config.get("trainable", True),
                )
            )
        elif class_name == "MaxPooling2D":
            model.add(
                tf.keras.layers.MaxPooling2D(
                    pool_size=tuple(config.get("pool_size", [2, 2])),
                    strides=tuple(config.get("strides", [2, 2])),
                    padding=config.get("padding", "valid"),
                    data_format=config.get("data_format", "channels_last"),
                    name=config.get("name"),
                )
            )
        elif class_name == "Flatten":
            model.add(
                tf.keras.layers.Flatten(
                    data_format=config.get("data_format", "channels_last"),
                    name=config.get("name"),
                    dtype=deserialize_dtype(config.get("dtype")),
                )
            )
        elif class_name == "Dense":
            model.add(
                tf.keras.layers.Dense(
                    units=config["units"],
                    activation=config.get("activation"),
                    use_bias=config.get("use_bias", True),
                    kernel_initializer=deserialize_initializer(
                        config.get("kernel_initializer")
                    ),
                    bias_initializer=deserialize_initializer(
                        config.get("bias_initializer")
                    ),
                    kernel_regularizer=tf.keras.regularizers.deserialize(
                        config.get("kernel_regularizer")
                    ),
                    bias_regularizer=tf.keras.regularizers.deserialize(
                        config.get("bias_regularizer")
                    ),
                    kernel_constraint=tf.keras.constraints.deserialize(
                        config.get("kernel_constraint")
                    ),
                    bias_constraint=tf.keras.constraints.deserialize(
                        config.get("bias_constraint")
                    ),
                    name=config.get("name"),
                    dtype=deserialize_dtype(config.get("dtype")),
                    trainable=config.get("trainable", True),
                )
            )
        else:
            raise ValueError(f"Unsupported layer type in legacy config: {class_name}")

    build_input_shape = sequential_config.get("build_input_shape")
    if build_input_shape:
        model.build(tuple(build_input_shape))

    return model


def deserialize_dtype(dtype_config):
    if isinstance(dtype_config, dict):
        config = dtype_config.get("config", {})
        return config.get("name")
    return dtype_config


def deserialize_initializer(initializer_config):
    if initializer_config is None:
        return None
    return tf.keras.initializers.deserialize(initializer_config)


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
    if MODEL_PATH.exists():
        st.write(f"Model file: `{MODEL_PATH.name}`")
        st.write("Model source: local project file")
    elif get_remote_model_url():
        st.write(f"Model file: `{MODEL_PATH.name}`")
        st.write("Model source: remote download via `MODEL_URL`")
    else:
        st.write("Model source is not configured yet.")
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
