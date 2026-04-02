# CNN Leaf Disease Classifier

This project is a Streamlit-based web application that classifies plant leaf diseases from uploaded images using a TensorFlow CNN model.

The app accepts a leaf image, preprocesses it to `224 x 224`, runs inference with a trained `.h5` model, and returns:

- the most likely disease class
- the prediction confidence
- the top 5 class probabilities

## Project Summary

This repository is focused on inference and deployment, not model training. The main deliverable is an interactive app for plant disease prediction.

Core stack:

- `Streamlit` for the UI
- `TensorFlow` for loading and running the CNN model
- `Pillow` and `NumPy` for image preprocessing
- `h5py` for model-file validation and legacy loading support
- `Docker` and `Heroku-style` config for deployment

The classifier currently supports `38` classes across these crop groups:

- Apple: 4 classes
- Blueberry: 1 class
- Cherry: 2 classes
- Corn: 4 classes
- Grape: 4 classes
- Orange: 1 class
- Peach: 2 classes
- Pepper: 2 classes
- Potato: 3 classes
- Raspberry: 1 class
- Soybean: 1 class
- Squash: 1 class
- Strawberry: 2 classes
- Tomato: 10 classes

The exact class mapping is stored in [`app/class_indices.json`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/app/class_indices.json).

## How The App Works

1. A user uploads a `.jpg`, `.jpeg`, or `.png` leaf image.
2. The image is converted to RGB, resized to `224 x 224`, normalized to `[0, 1]`, and expanded into a batch.
3. The app loads the trained model from:
   - a local file at `app/trained_model/plant_disease_prediction_model.h5`, or
   - a remote URL provided through `MODEL_URL`
4. TensorFlow runs inference on the image.
5. The app shows:
   - the predicted class
   - overall confidence
   - a top-5 confidence table

## Folder And File Analysis

### Root Directory

| Path | Purpose | Notes |
| --- | --- | --- |
| [`README.md`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/README.md) | Project documentation | Rewritten to reflect the actual codebase and deployment flow. |
| [`.gitignore`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/.gitignore) | Ignore rules | Excludes `.venv`, `__pycache__`, Streamlit secrets, and `.h5` model artifacts from Git. |
| [`requirements.txt`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/requirements.txt) | Python dependencies | Pins app runtime packages. |
| [`Dockerfile`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/Dockerfile) | Main container build file | This is the primary Docker configuration for the project. |
| [`Procfile`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/Procfile) | Process startup definition | Launches Streamlit for platforms that use Procfiles. |
| [`heroku.yml`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/heroku.yml) | Heroku container config | Points Heroku to the root `Dockerfile`. |
| [`runtime.txt`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/runtime.txt) | Python runtime version | Declares `python-3.10.14`. |
| [`test_apple_black_rot.JPG`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/test_apple_black_rot.JPG) | Sample test image | Useful for quick manual prediction checks. |
| [`test_blueberry_healthy.jpg`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/test_blueberry_healthy.jpg) | Sample test image | Useful for quick manual prediction checks. |
| [`test_potato_early_blight.jpg`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/test_potato_early_blight.jpg) | Sample test image | Useful for quick manual prediction checks. |
| [`.streamlit/`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/.streamlit) | Active local Streamlit config | This is the standard place Streamlit reads config from during local runs. |
| [`.venv/`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/.venv) | Local virtual environment | Development-only environment, not part of app logic. |
| [`app/`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/app) | Main application package | Contains the Streamlit app, metadata, and model assets. |

### `.streamlit`

| Path | Purpose | Notes |
| --- | --- | --- |
| [`.streamlit/config.toml`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/.streamlit/config.toml) | Local Streamlit runtime config | Sets `headless=true`, disables CORS and XSRF protection, and disables usage stats. |

### `app`

| Path | Purpose | Notes |
| --- | --- | --- |
| [`app/main.py`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/app/main.py) | Main Streamlit application | Contains UI, image preprocessing, model loading, prediction, remote download support, and legacy model compatibility logic. |
| [`app/class_indices.json`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/app/class_indices.json) | Class index to class label mapping | Used to turn model output indices into readable disease labels. |
| [`app/config.toml`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/app/config.toml) | Alternate Streamlit config | Appears to be an older deployment config that targets port `80`. |
| [`app/dockerFile`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/app/dockerFile) | Older Docker setup | Likely legacy. It refers to `credentials.toml`, which is not present in this repo. |
| [`app/trained_model/`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/app/trained_model) | Model assets folder | Contains the trained model file and a text pointer for remote hosting. |
| [`app/__pycache__/`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/app/__pycache__) | Python bytecode cache | Auto-generated and not important for source control or documentation. |

### `app/trained_model`

| Path | Purpose | Notes |
| --- | --- | --- |
| [`app/trained_model/plant_disease_prediction_model.h5`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/app/trained_model/plant_disease_prediction_model.h5) | Trained CNN model artifact | Present in the current workspace and approximately `547 MB`; ignored by Git via `.gitignore`. |
| [`app/trained_model/model.text`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/app/trained_model/model.text) | Remote model URL example | Placeholder text showing how a hosted model URL could be defined. |

## Main Application Analysis

The core application lives in [`app/main.py`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/app/main.py).

Important responsibilities inside this file:

- sets up the Streamlit page and upload UI
- loads the model with `@st.cache_resource`
- loads class names with `@st.cache_data`
- supports both local model loading and remote model download
- validates downloaded model files before using them
- includes fallback logic for legacy `.h5` models that contain `quantization_config`
- preprocesses images into the expected CNN input format
- returns the top prediction and the top 5 confidence scores

### Model Loading Strategy

The model loading path is more robust than a basic TensorFlow script:

- If the local model file exists, the app uses it directly.
- If it does not exist, the app looks for `MODEL_URL` in:
  - Streamlit secrets
  - environment variables
- If the remote source is protected, the app can also use `MODEL_DOWNLOAD_TOKEN`.
- Downloaded files are cached in the system temp directory.
- The downloaded file is validated with `h5py` before inference begins.

### Legacy Compatibility

The app includes custom logic to rebuild older Sequential TensorFlow models if standard loading fails because of legacy quantization metadata. That is a useful reliability feature for deployment, especially when model artifacts come from different training environments.

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Make sure a model is available

Choose one of these approaches:

- Place `plant_disease_prediction_model.h5` inside `app/trained_model/`
- Or configure `MODEL_URL` so the app can download it at startup

Optional protected-host token:

```bash
export MODEL_DOWNLOAD_TOKEN="your_access_token"
```

Remote model URL example:

```bash
export MODEL_URL="https://huggingface.co/your-user/your-repo/resolve/main/plant_disease_prediction_model.h5"
```

## Running The App Locally

```bash
streamlit run app/main.py
```

Then open the local URL shown by Streamlit in your terminal.

## Docker

Build:

```bash
docker build -t cnn-leaf-classifier .
```

Run:

```bash
docker run -p 8501:8501 cnn-leaf-classifier
```

The root [`Dockerfile`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/Dockerfile) starts Streamlit with:

```bash
streamlit run app/main.py --server.port=${PORT:-8501} --server.address=0.0.0.0
```

## Deployment Notes

This repo includes multiple deployment-related files:

- [`Dockerfile`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/Dockerfile): primary container build file
- [`Procfile`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/Procfile): process startup command
- [`heroku.yml`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/heroku.yml): Heroku container deploy config
- [`runtime.txt`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/runtime.txt): Python version pin

These are consistent with a deployed inference app, but there is also an older file:

- [`app/dockerFile`](/Users/shreyashgolhani/Documents/CNN_leaf_classifer/app/dockerFile) appears to be a legacy Docker setup and references a missing `credentials.toml`

## Sample Inputs

You already have three test images in the root directory:

- apple black rot
- blueberry healthy
- potato early blight

They are useful for manual smoke testing after startup.

## Current Strengths

- simple and usable Streamlit interface
- support for both local and remote model files
- pinned dependency versions
- Docker and platform deployment support
- fallback support for older TensorFlow `.h5` model formats
- sample images included for quick validation

## Current Gaps And Observations

- there is no training pipeline in this repository
- there are no automated tests yet
- `app/config.toml` and `.streamlit/config.toml` overlap in purpose
- `app/dockerFile` looks outdated compared with the root `Dockerfile`
- the main model file is very large and intentionally excluded from Git

## Recommended Cleanup

If you want to make the repo easier to maintain, the next sensible cleanup steps would be:

1. Keep the root `Dockerfile` as the single Docker source of truth.
2. Remove or archive `app/dockerFile` if it is no longer used.
3. Decide whether `.streamlit/config.toml` or `app/config.toml` is the real config to keep.
4. Add a short test script or smoke test for startup and prediction.
5. Add training details only if you plan to include model-building code in this repository.

## Verification

The application file was syntax-checked successfully with:

```bash
python3 -m py_compile app/main.py
```

No automated tests were found in the repository at the time of this documentation update.
