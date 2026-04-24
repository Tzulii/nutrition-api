from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import tflite_runtime.interpreter as tflite  # lighter than tensorflow

app = Flask(__name__)

# -----------------------------
# Load TFLite model
# -----------------------------
interpreter = tflite.Interpreter(model_path="food101_model.tflite")
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
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

# -----------------------------
# Root route (IMPORTANT for Render)
# -----------------------------
@app.route("/")
def home():
    return "API is running"

# -----------------------------
# Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # Parse JSON safely
    data = request.get_json(force=True)

    # Validate input
    if not data or "image" not in data:
        return jsonify({
            "error": "Invalid input. JSON must contain 'image' key."
        }), 400

    # Decode base64 image
    try:
        img_str = data["image"]

        # Remove header if exists
        if "," in img_str:
            img_str = img_str.split(",")[1]

        img_data = base64.b64decode(img_str)
        image = Image.open(BytesIO(img_data)).convert("RGB")
    except Exception as e:
        return jsonify({
            "error": "Invalid image data",
            "details": str(e)
        }), 400

    # Preprocess image
    try:
        input_data = preprocess(image)
    except Exception as e:
        return jsonify({
            "error": "Preprocessing failed",
            "details": str(e)
        }), 500

    # Run inference
    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
    except Exception as e:
        return jsonify({
            "error": "Model inference failed",
            "details": str(e)
        }), 500

    # Get prediction
    try:
        index = int(np.argmax(output))
        label = labels[index]
        confidence = float(output[0][index])
    except Exception as e:
        return jsonify({
            "error": "Post-processing failed",
            "details": str(e)
        }), 500

    # Return result
    return jsonify({
        "prediction": label,
        "confidence": confidence
    })

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)