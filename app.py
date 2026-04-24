from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import tensorflow as tf

app = Flask(__name__)

# -----------------------------
# Load TensorFlow Lite model (TensorFlow version)
# -----------------------------
interpreter = tf.lite.Interpreter(model_path="food101_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# Load labels
# -----------------------------
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Root route
# -----------------------------
@app.route("/")
def home():
    return "API is running"

# -----------------------------
# Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(force=True)

    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' key"}), 400

    try:
        img_str = data["image"]

        if "," in img_str:
            img_str = img_str.split(",")[1]

        img_data = base64.b64decode(img_str)
        image = Image.open(BytesIO(img_data)).convert("RGB")

    except Exception as e:
        return jsonify({"error": "Invalid image", "details": str(e)}), 400

    try:
        input_data = preprocess(image)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])

        index = int(np.argmax(output))
        label = labels[index]
        confidence = float(output[0][index])

    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })

# -----------------------------
# Run server (Render-safe)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)