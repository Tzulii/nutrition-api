from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import os
import tensorflow as tf

app = Flask(__name__)

# -----------------------------
# Lazy load TFLite model (IMPORTANT for Render stability)
# -----------------------------
interpreter = None

def get_interpreter():
    global interpreter
    if interpreter is None:
        interpreter = tf.lite.Interpreter(model_path="food101_model.tflite")
        interpreter.allocate_tensors()
    return interpreter

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
    return np.expand_dims(image, axis=0)

# -----------------------------
# Health check route
# -----------------------------
@app.route("/")
def home():
    return "API is running"

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(force=True)

        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' key"}), 400

        img_str = data["image"]

        # remove base64 header if exists
        if "," in img_str:
            img_str = img_str.split(",")[1]

        img_data = base64.b64decode(img_str)
        image = Image.open(BytesIO(img_data)).convert("RGB")

        input_data = preprocess(image)

        interpreter = get_interpreter()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])

        index = int(np.argmax(output))
        label = labels[index]
        confidence = float(output[0][index])

        return jsonify({
            "prediction": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "error": "Server error",
            "details": str(e)
        }), 500


# -----------------------------
# Render-safe run
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)