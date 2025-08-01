from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load your model and labels outside the route, once
model = tf.keras.models.load_model('converted_keras/keras_model.h5')
with open("converted_keras/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess_image(image_bytes):
    # Try opening the image with PIL to verify
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image: {e}")

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image_base64' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image_base64 = data['image_base64']

    # Remove data URI prefix if present
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    try:
        # Decode base64 to bytes
        image_data = base64.b64decode(image_base64)

        # Verify image can be opened (to catch errors early)
        image = preprocess_image(image_data)

        prediction = model.predict(image)

        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        predicted_label = labels[class_id]

        return jsonify({
            'class_id': class_id,
            'label': predicted_label,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
