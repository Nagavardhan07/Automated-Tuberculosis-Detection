from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load your trained models
capsnet_model = load_model(r"C:\Users\nagavardhan\OneDrive\Desktop\0709\models\capsnet_feature_model.h5")
densenet_model = DenseNet121(weights="imagenet", include_top=False, pooling="avg")

# Create a folder for uploads if it doesn't exist
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Feature extraction
def extract_features_from_image(img_array):
    features = densenet_model.predict(img_array)
    return features

# Prediction function
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    features = extract_features_from_image(img_array)
    
    prediction = capsnet_model.predict(features)
    probability = prediction[0][0]
    
    threshold = 0.7
    if probability > threshold:
        result = "TB Negative"
        confidence = probability * 100
    else:
        result = "TB Positive"
        confidence = (1 - probability) * 100
    
    return result, probability, confidence

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))  # Go back to home

    file = request.files['image']  # <--- IMPORTANT: name="image" in your HTML form

    if file.filename == '':
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Perform prediction
    result, probability, confidence = predict_image(filepath)

    return render_template('result.html', prediction=result, probability=round(probability, 4), confidence=round(confidence, 2), filepath=filepath)

if __name__ == "__main__":
    app.run(debug=True)
