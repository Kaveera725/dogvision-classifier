import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF logging
from pathlib import Path

def print_usage():
    print("Usage:")
    print("python src/predict.py <image_path>")

def main():
    if len(sys.argv) < 2:
        print("Error: Missing image path.")
        print_usage()
        sys.exit(1)
        
    image_path_str = sys.argv[1]
    image_path = Path(image_path_str)
    
    if not image_path.exists():
        print(f"Error: The file '{image_path_str}' does not exist.")
        sys.exit(1)
        
    if not image_path.is_file():
        print(f"Error: '{image_path_str}' is not a file.")
        sys.exit(1)
        
    # Check if model exists
    model_path = Path("models/cat_dog_classifier.keras")
    if not model_path.exists():
        print("Error: Model file 'models/cat_dog_classifier.keras' not found.")
        sys.exit(1)
        
    try:
        import tensorflow as tf
        import numpy as np
    except ImportError:
        print("Error: Required libraries (tensorflow, numpy) are not installed.")
        sys.exit(1)

    # Validate and load image
    try:
        # load_img handles PIL conversion to RGB automatically
        img = tf.keras.utils.load_img(image_path, target_size=(128, 128))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error: Could not read or process image '{image_path_str}'. It might be corrupted or in an unsupported format.")
        sys.exit(1)

    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print("Error: Failed to load the model.")
        sys.exit(1)

    # Generate prediction
    try:
        prediction_prob = model.predict(img_array, verbose=0)[0][0]
    except Exception as e:
        print("Error: Prediction failed.")
        sys.exit(1)

    if prediction_prob >= 0.5:
        prediction = "DOG"
        confidence = prediction_prob * 100
    else:
        prediction = "CAT"
        confidence = (1.0 - prediction_prob) * 100

    print("========================================")
    print("CAT vs DOG PREDICTION")
    print("========================================")
    print()
    print(f"Image: {image_path_str}")
    print()
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    print()
    print("========================================")

if __name__ == "__main__":
    main()
