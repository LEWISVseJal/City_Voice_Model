# City Voice

City Voice is an AI-powered civic issue detection web app. It accepts an uploaded image and classifies visible public-infrastructure issues such as road damage, potholes, garbage, waterlogging, open manholes, fallen trees, and street-light problems.

The app uses a Flask frontend/API with an ONNX model for inference. Training scripts are included for a ResNet18-based classifier and a ResNet18 + CBAM variant.

## Features

- Upload an image through a simple web interface.
- Predict one of 12 civic-image classes.
- Show prediction confidence and a short issue description.
- Run ONNX inference with `onnxruntime`.
- Use zoom/crop voting during prediction to improve detection of small or localized issues.
- Train and convert PyTorch models to ONNX.

## Classes

The current classifier supports these classes:

- `clean_dustbin`
- `crack_road`
- `fallen_trees`
- `garbage`
- `irrelevant`
- `open_manholes`
- `plain_road`
- `pothole_road`
- `road_damage`
- `street_light`
- `waterlogging`
- `working_streetlight`

Issue classes used by the app include:

- `crack_road`
- `pothole_road`
- `road_damage`
- `fallen_trees`
- `garbage`
- `open_manholes`
- `waterlogging`
- `street_light`

## Project Structure

```text
City_Voice/
|-- api.py                    # Flask app and ONNX inference pipeline
|-- templates/index.html      # Upload and prediction result page
|-- requirements.txt          # Runtime dependencies for the Flask app
|-- Procfile                  # Gunicorn command for deployment
|-- runtime.txt               # Python runtime version
|-- convertnew_to_onnx.py     # Converts CBAM PyTorch model to ONNX
|-- split_dataset.py          # Creates train/validation split
|-- data.yaml                 # YOLO dataset config for pothole/crack data
|-- dataset/                  # Original dataset
|-- dataset_new/              # Split/balanced dataset
|-- hard_negative_samples/    # Non-issue examples for rejection testing
`-- uploads/                  # Upload-related local folder
```

## Requirements

The web app runtime dependencies are listed in `requirements.txt`:

```text
flask
flask-cors
onnxruntime
pillow
gunicorn
numpy
```

Training requires additional packages that are not listed in the runtime file:

```text
torch
torchvision
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

For training or model conversion, also install PyTorch and TorchVision:

```bash
pip install torch torchvision
```

## Running the Web App

Make sure the ONNX model file exists in the project root:

```text
city_voice_cbam.onnx
```

Start the Flask app:

```bash
python api.py
```

Open the app in your browser:

```text
http://localhost:7063
```

Upload an image and submit it to receive the predicted category, confidence score, and description.

## Prediction Endpoint

The app exposes a `POST /predict` endpoint that expects a multipart form upload using the field name `file`.

Example:

```bash
curl -X POST http://localhost:7063/predict -F "file=@sample.jpg"
```

The current endpoint renders `templates/index.html` with the prediction result. Error responses are returned as JSON.

## Dataset Layout

The training script expects this dataset layout:

```text
dataset_new/
|-- train_balanced/
|   |-- clean_dustbin/
|   |-- crack_road/
|   `-- ...
`-- val/
    |-- clean_dustbin/
    |-- crack_road/
    `-- ...
```

Use `split_dataset.py` to create an 80/20 split from `dataset/train` into `dataset_new/train` and `dataset_new/val`:

```bash
python split_dataset.py
```

The script uses:

- ResNet18 pretrained on ImageNet.
- A custom final classifier for 12 classes.
- Data augmentation for training images.
- Clean validation transforms.
- Class-weighted cross-entropy with label smoothing.
- AdamW optimizer.
- Cosine annealing learning-rate scheduler.

The model is saved as:

```text
city_voice_cbam_best.pth
```

## ONNX Conversion

Convert the trained CBAM model to ONNX:

```bash
python convertnew_to_onnx.py
```

This reads:

```text
city_voice_cbam_best.pth
```

and writes:

```text
city_voice_cbam.onnx
```

The Flask app loads `city_voice_cbam.onnx` by default.

## Deployment

The included `Procfile` runs the app with Gunicorn:

```text
web: gunicorn api:app --workers 1 --timeout 120 --worker-class sync
```

The Python runtime is set in `runtime.txt`:

```text
python-3.10.0
```
