from flask import Flask, request, send_file, jsonify, render_template
from PIL import Image, ImageChops, ImageEnhance
from tensorflow import keras
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import io
import base64
import os

app = Flask(__name__)

# Load the trained model when the application starts
model = load_model("C:\\Users\\Sanjivkumar Naik\\Desktop\\myflaskapp\\ela.h5")

@app.route('/')
def index():
    return render_template('ForgeryDetection.html')

def convert_to_ela_image(image_path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'ela_image.jpg'

    image = Image.open(image_path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    ela_image.save(ela_filename, 'JPEG')

    return ela_filename

@app.route('/process_image', methods=['POST'])
def process_image():
    file = request.files['image_data']
    method = request.form['method']
    image_path = 'input_image.jpg'
    file.save(image_path)

    ela_image_path = convert_to_ela_image(image_path, quality=90)

    # Preprocess the image for the model
    ela_image = Image.open(ela_image_path)
    x = image.img_to_array(ela_image.resize((224, 224)))
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    # Make a prediction using the preloaded model
    prediction = model.predict(x)

    # Prepare the prediction result
    if prediction > 0.5:
        prediction_result = "The image is predicted to be original."
    else:
        prediction_result = "The image is predicted to be forged."

    # Convert ELA image to a base64 string
    with open(ela_image_path, "rb") as image_file:
        ela_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    # Delete the saved images
    # os.remove(image_path)
    # os.remove(ela_image_path)

    # Return a JSON response with the ELA image and the prediction result
    return jsonify({'ela_image': 'data:image/jpeg;base64,' + ela_image_base64, 'prediction_result': prediction_result})

if __name__ == '__main__':
    app.run(debug=True)
