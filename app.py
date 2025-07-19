from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
model = tf.keras.models.load_model('models/plant_leaf_model.h5')

# Define class names (update based on your dataset)
class_names = [
    'aloevera', 'banana', 'bilimbi', 'cantaloupe', 'cassava', 'coconut',
    'corn', 'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger', 'guava',
    'kale', 'longbeans', 'mango', 'melon', 'orange', 'paddy', 'papaya',
    'peper chili', 'pineapple', 'pomelo', 'shallot', 'soybeans'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if not file:
            return "No file uploaded", 400

        # Create a unique filename and save image
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Read and preprocess the image
        img = cv2.imread(filepath)
        img_resized = cv2.resize(img, (224, 224)) / 255.0
        img_input = np.expand_dims(img_resized, axis=0)

        # Predict using the model
        prediction = model.predict(img_input)
        predicted_class = class_names[np.argmax(prediction)]

        # Pass prediction and image path to result template
        return render_template('result.html', prediction=predicted_class, image_path=filepath)
    
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
