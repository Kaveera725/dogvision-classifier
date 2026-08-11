import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from pathlib import Path
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max upload

# Load models at startup
MODEL_PATH = Path("models/mobilenetv2_cat_dog_classifier.keras")
print("Loading custom model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Custom model loaded successfully!")
except Exception as e:
    print(f"Error loading custom model: {e}")
    model = None

print("Loading validation model (MobileNetV2)...")
try:
    validation_model = tf.keras.applications.MobileNetV2(weights='imagenet')
    print("Validation model loaded successfully!")
except Exception as e:
    print(f"Error loading validation model: {e}")
    validation_model = None

def is_cat_or_dog(filepath):
    """Uses MobileNetV2 to check if the image contains a cat or a dog."""
    if validation_model is None:
        return True # Fallback if model fails to load
    
    img = tf.keras.utils.load_img(filepath, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    predictions = validation_model.predict(img_array, verbose=0)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]
    
    # Check if any of the top 5 predictions contain cat, dog, or related keywords
    keywords = ['cat', 'dog', 'hound', 'puppy', 'kitten', 'terrier', 'spaniel', 'retriever', 'collie', 'husky', 'pug', 'chihuahua']
    
    for _, label, _ in decoded_preds:
        label_lower = label.lower()
        if any(keyword in label_lower for keyword in keywords):
            return True
            
    return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part.'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Validate image first
            if not is_cat_or_dog(filepath):
                os.remove(filepath) # Clean up
                return jsonify({'error': 'This image does not appear to contain a cat or a dog. Please upload a valid image.'}), 400

            # Process image for our model
            img = tf.keras.utils.load_img(filepath, target_size=(128, 128))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction_prob = model.predict(img_array, verbose=0)[0][0]
            
            if prediction_prob >= 0.5:
                prediction_class = "Dog"
                confidence = float(prediction_prob * 100)
            else:
                prediction_class = "Cat"
                confidence = float((1.0 - prediction_prob) * 100)
                
            if confidence < 80.0:
                return jsonify({'error': f"Cannot confidently detect a cat or dog. Confidence was only {confidence:.2f}%."}), 400
                
            return jsonify({
                'prediction': prediction_class,
                'confidence': f"{confidence:.2f}%",
                'image_url': f"/{filepath}"
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type.'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
