import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model('models/plant_leaf_model.h5')

# Define the class names (must match the folder names used during training)
class_names = [
    'aloevera', 'banana', 'bilimbi', 'cantaloupe', 'cassava', 'coconut', 'corn', 
    'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger', 'guava', 'kale', 
    'longbeans', 'mango', 'melon', 'orange', 'paddy', 'papaya', 'peper chili', 
    'pineapple', 'pomelo', 'shallot', 'soybeans'
]

# Function to preprocess the input image
def preprocess_image(image_path, img_size=(224, 224)):
    if not os.path.exists(image_path):
        print("Error: Image path does not exist.")
        return None

    try:
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Unable to read the image. Please check the file format.")
            return None
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

# Function to predict the class of the plant leaf
def predict_plant(image_path):
    img = preprocess_image(image_path)
    if img is not None:
        try:
            prediction = model.predict(img)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100  # Confidence in percentage
            return predicted_class, confidence
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, None
    else:
        return None, None

# Main execution block
if __name__ == "__main__":
    # Get the image path from the user
    image_path = input("C:\\Users\Muzammil Hameed\\Desktop\\plant_leaf_detection\\uploads\\aloevera5.jpg):").strip()

    # Check if the file exists and predict
    if os.path.exists(image_path):
        plant_name, confidence = predict_plant(image_path)
        if plant_name:
            print(f"\nPredicted Plant: {plant_name}")
            print(f"Confidence: {confidence:.2f}%")
        else:
            print("Prediction failed. Please ensure the input image is valid.")
    else:
        print("The specified image path does not exist. Please check and try again.")
