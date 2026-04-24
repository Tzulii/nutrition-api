from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import tensorflow as tf

app = Flask(__name__)

# -----------------------------
# Load TFLite model (TensorFlow runtime)
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
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Health check (Render required)
# -----------------------------
@app.route("/")
def home():
    return "Nutrition API is running"

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({
                "error": "Missing 'image' key"
            }), 400

        # Decode base64 image
        img_str = data["image"]

        if "," in img_str:
            img_str = img_str.split(",")[1]

        img_data = base64.b64decode(img_str)
        image = Image.open(BytesIO(img_data)).convert("RGB")

        # Preprocess
        input_data = preprocess(image)

        # Run inference
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        # Get result
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
# Run (Render uses gunicorn, this is fallback only)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)