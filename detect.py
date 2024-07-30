from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Load the model
model_path = 'model1_vgg16.h5'  # Change to the appropriate path on your system
model = load_model(model_path)

# Function to process uploaded image
def process_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    return x

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Result page
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        # If file is available and valid, process it
        if file:
            # Save the uploaded image
            img_path = 'uploaded_image.jpg'  # Change to the appropriate path on your system
            file.save(img_path)

            # Process the image
            x = process_image(img_path)

            # Make predictions
            predictions = model.predict(x)

            # Interpret the predictions
            print(predictions)
            fake_probability = predictions[0][0]  # Assuming first index represents the probability of being fake
            is_fake = fake_probability > 0.5  # You can adjust the threshold as per your requirement

            # Display the image with the prediction
            plt.imshow(image.load_img(img_path))
            '''if is_fake:
                plt.title(f"Fake (Probability: {fake_probability:.2f})")
            else:
                plt.title(f"Real (Probability: {1 - fake_probability:.2f})")'''
            plt.axis('off')

            # Save the plot as an image file
            plot_path = 'static/plot.png'
            plt.savefig(plot_path)
            plt.close()  # Close the plot to free up resources

            # Convert plot to base64 for embedding in HTML
            with open(plot_path, "rb") as img_file:
                encoded_plot = base64.b64encode(img_file.read()).decode('utf-8')

            # Determine the result
            if is_fake:
                Result = "Provided Logo is Fake"
            else:
                Result = "Provided Logo is Original"

            # Pass the base64-encoded plot and result to the result template
            return render_template('result.html', prediction=Result, image=encoded_plot)

if __name__ == '__main__':
    app.run()
