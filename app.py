from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('keras_model.h5')

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Image preprocessing
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['file']
    image = preprocess_image(file.read())
    prediction = model.predict(image)
    
    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    predicted_label = labels[class_id]

    return jsonify({
        'class_id': class_id,
        'label': predicted_label,
        'confidence': round(confidence, 2)
    })

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
