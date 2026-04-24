from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import tensorflow as tf

app = Flask(__name__)

# =============================
# LOAD MODEL (RENDER SAFE)
# =============================
model = tf.lite.Interpreter(model_path="food101_model.tflite")
model.allocate_tensors()

input_details = model.get_input_details()
output_details = model.get_output_details()

# =============================
# LOAD LABELS
# =============================
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# =============================
# PREPROCESS IMAGE
# =============================
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =============================
# HOME ROUTE
# =============================
@app.route("/")
def home():
    return "API is running on Render 🚀"

# =============================
# PREDICTION ROUTE
# =============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        img_str = data["image"]

        # remove base64 header if exists
        if "," in img_str:
            img_str = img_str.split(",")[1]

        img_data = base64.b64decode(img_str)
        image = Image.open(BytesIO(img_data)).convert("RGB")

        input_data = preprocess(image)

        # run inference
        model.set_tensor(input_details[0]["index"], input_data)
        model.invoke()

        output = model.get_tensor(output_details[0]["index"])

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


# =============================
# RENDER ENTRY POINT
# =============================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)