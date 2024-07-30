from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

# Keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model_resnet.h5'

# Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Wrap the model's predict function using tf.function
@tf.function
def model_predict(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    # Preprocess the input (this step might vary depending on how the model was trained)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    # Make prediction
    predictions = model(img_array)
    return predictions

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)

        # Process the result
        pred_class = decode_predictions(preds, top=1)[0][0]
        result = pred_class[1]  # Get the class label
        return result

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
