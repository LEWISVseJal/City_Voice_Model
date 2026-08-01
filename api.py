import base64
import gc
import io
import os
from collections import Counter

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = "city_voice_cbam.onnx"

classes = [
    "clean_dustbin", "crack_road", "fallen_trees", "garbage",
    "irrelevant", "open_manholes", "plain_road", "pothole_road",
    "road_damage", "street_light", "waterlogging", "working_streetlight"
]

CLASS_DESCRIPTIONS = {
    "crack_road": "Cracks visible on road surface. May expand and cause serious damage.",
    "pothole_road": "Pothole detected. Safety hazard for vehicles and pedestrians.",
    "garbage": "Garbage present in a public area. Affects hygiene and cleanliness.",
    "street_light": "Street light appears non-functional or damaged.",
    "road_damage": "Visible structural road damage affecting vehicle movement.",
    "open_manholes": "Open manhole detected. Serious safety risk.",
    "waterlogging": "Waterlogging detected. May indicate drainage issues.",
    "fallen_trees": "Fallen tree on road or public area. Blocks traffic.",
    "clean_dustbin": "Dustbin is clean and maintained.",
    "plain_road": "Road appears in good condition.",
    "working_streetlight": "Street light is functional.",
    "irrelevant": "No civic issue detected in this image."
}

ISSUE_CLASSES = [
    "crack_road",
    "pothole_road",
    "road_damage",
    "fallen_trees",
    "garbage",
    "open_manholes",
    "waterlogging",
    "street_light"
]

if os.path.exists(MODEL_PATH):
    session = ort.InferenceSession(MODEL_PATH)
    print("✅ ONNX Model loaded:", MODEL_PATH)
else:
    session = None
    print("⚠️ Model not found:", MODEL_PATH)


@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return response


@app.route("/")
def home():
    return render_template("index.html")


def preprocess_pil(image):
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_array = (img_array - mean) / std
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    return img_array


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def crop_box(image, name, left, top, crop_size):
    w, h = image.size

    left = max(0, min(left, w - crop_size))
    top = max(0, min(top, h - crop_size))

    return (
        name,
        image.crop((left, top, left + crop_size, top + crop_size))
    )


def get_zoom_crops(image):
    w, h = image.size
    crops = []

    min_side = min(w, h)

    crops.append(("full_image", image))

    # Center zoom crops
    for scale in [0.85, 0.75, 0.65, 0.55]:
        crop_size = int(min_side * scale)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        crops.append(crop_box(image, f"center_zoom_{int(scale * 100)}", left, top, crop_size))

    # Lower crops useful for potholes / road damage
    for scale in [0.75, 0.65, 0.55]:
        crop_size = int(min_side * scale)
        left = (w - crop_size) // 2
        top = int(h * 0.35)
        crops.append(crop_box(image, f"lower_zoom_{int(scale * 100)}", left, top, crop_size))

    # Upper crops useful for trees / poles / street lights
    for scale in [0.75, 0.65]:
        crop_size = int(min_side * scale)
        left = (w - crop_size) // 2
        top = int(h * 0.08)
        crops.append(crop_box(image, f"upper_zoom_{int(scale * 100)}", left, top, crop_size))

    # Left and right crops useful for fallen trees on side
    for scale in [0.75, 0.65]:
        crop_size = int(min_side * scale)

        left = int(w * 0.05)
        top = (h - crop_size) // 2
        crops.append(crop_box(image, f"left_zoom_{int(scale * 100)}", left, top, crop_size))

        left = int(w * 0.95) - crop_size
        top = (h - crop_size) // 2
        crops.append(crop_box(image, f"right_zoom_{int(scale * 100)}", left, top, crop_size))

    # Lower-right crop for fallen trees / road obstruction
    for scale in [0.70, 0.60]:
        crop_size = int(min_side * scale)
        left = int(w * 0.65) - crop_size // 2
        top = int(h * 0.55) - crop_size // 2
        crops.append(crop_box(image, f"lower_right_zoom_{int(scale * 100)}", left, top, crop_size))

    return crops


def predict_single_image(image):
    img_array = preprocess_pil(image)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})

    logits = outputs[0][0]
    probs = softmax(logits)

    pred = int(np.argmax(probs))
    conf = float(probs[pred]) * 100

    return pred, conf, probs


def predict_with_zoom(image):
    crops = get_zoom_crops(image)

    votes = []
    confidences = {}
    probs_by_class = {}
    top3_all_crops = []

    print("\n========== ZOOM PREDICTIONS ==========")

    for crop_name, crop_img in crops:
        pred, conf, probs = predict_single_image(crop_img)
        class_name = classes[pred]

        top3_idx = np.argsort(probs)[::-1][:3]
        top3_names = [classes[i] for i in top3_idx]
        top3_all_crops.extend(top3_names)

        print(crop_name, "=>", class_name, ":", round(conf, 2), "%", "| Top3:", top3_names)

        votes.append(class_name)
        confidences.setdefault(class_name, []).append(conf)
        probs_by_class[class_name] = probs

    print("======================================")

    vote_counter = Counter(votes)
    top3_counter = Counter(top3_all_crops)

    print("Votes:", vote_counter)
    print("Top3 Mentions:", top3_counter)

    max_votes = max(vote_counter.values())
    tied_classes = [
        cls for cls, count in vote_counter.items()
        if count == max_votes
    ]

    if len(tied_classes) == 1:
        final_class = tied_classes[0]
    else:
        print("Tie detected:", tied_classes)

        final_class = max(
            tied_classes,
            key=lambda cls: sum(confidences[cls]) / len(confidences[cls])
        )

    avg_conf = sum(confidences[final_class]) / len(confidences[final_class])
    final_probs = probs_by_class[final_class]

    # Rescue rule for fallen trees
    if final_class == "street_light" and avg_conf < 45:
        if top3_counter["fallen_trees"] >= 2:
            final_class = "fallen_trees"
            avg_conf = max(avg_conf, 35)
            final_probs = probs_by_class.get("fallen_trees", final_probs)
            print("Rescue Rule Applied: street_light changed to fallen_trees")

    print("Final Prediction:", final_class, "| Avg Confidence:", round(avg_conf, 2))

    return final_class, avg_conf, final_probs


@app.route("/predict", methods=["POST"])
def predict():
    gc.collect()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        if session is None:
            return jsonify({"error": "Model not loaded"}), 500

        file_bytes = file.read()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image_base64 = base64.b64encode(file_bytes).decode("utf-8")

        predicted_class, confidence, probs = predict_with_zoom(img)
        confidence = round(confidence, 2)

        top3_idx = np.argsort(probs)[::-1][:3]

        print("\n========== FINAL TOP 3 PREDICTIONS ==========")
        for idx in top3_idx:
            print(classes[idx], ":", round(float(probs[idx]) * 100, 2), "%")
        print("=============================================\n")

        if confidence < 30 and predicted_class not in ISSUE_CLASSES:
            predicted_class = "irrelevant"
            description = "Unable to confidently identify a civic issue."
        else:
            description = CLASS_DESCRIPTIONS.get(predicted_class, "Issue detected.")

        return render_template(
            "index.html",
            prediction=predicted_class,
            confidence=confidence,
            description=description,
            main_category=predicted_class,
            image_data=image_base64
        )

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7063, debug=False)
