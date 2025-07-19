import cv2
import os

# Define input and output directories
input_dir = 'C:/Users/Muzammil Hameed/Desktop/plant_leaf_detection/dataset/raw_images/'
output_dir = 'C:/Users/Muzammil Hameed/Desktop/plant_leaf_detection/dataset/preprocessed_images/'

size = (224, 224)  # Resize dimensions

def preprocess_images(input_dir, output_dir, size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        save_dir = os.path.join(output_dir, class_name)
        os.makedirs(save_dir, exist_ok=True)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, size)  # Resize image
                cv2.imwrite(os.path.join(save_dir, img_name), img)  # Save preprocessed image
                print(f"Processed {img_name} for {class_name}")

# Run the preprocessing function
if __name__ == '__main__':
    preprocess_images(input_dir, output_dir, size)
