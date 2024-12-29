# Fake Logo Detection

**APPROACH DESCRIPTION**
▪
Gathering Data : We initiated the project by collecting a diverse set of images
depicting both authentic and counterfeit logos. This dataset was compiled from Kaggle.

▪
Organizing Files : Following data collection, we meticulously organized the images
into distinct directories corresponding to real and fake logos. This step streamlines the
training phases, facilitating efficient data handling.

▪
Splitting Data : The dataset was split into two folders - one for training and another
for testing - to ensure proper evaluation of the model's performance.
▪
Training the Model : Leveraging the Keras library with TensorFlow backend, we
implemented the VGG16 architecture. We loaded the directory of images and trained
the model with training-testing split ratios to evaluate its performance.

▪
Building an Interface : We created a user-friendly interface using Flask to interact
with the trained model. Flask, a web development framework for Python, was employed
for this purpose. The interface will provide users with the functionality to upload
images of logos for authenticity assessment.

▪
Testing the Model : Once the training phase concluded, the model's accuracy was
assessed by testing it on a separate dataset, distinct from the one used for training. This
evaluation involved comparing the model's predictions with the actual labels of the
logos in the test dataset.

▪
Final Output : The final output of our system is a decision about the authenticity of
the logo, determining whether it is fake or real. Users interacting with the interface will
receive a decisive verdict regarding the authenticity of the uploaded logo images.

**DATA Logo dataset**

The fake logo detection dataset is structured similarly, with a primary folder containing two
distinct subfolders: "fake" and "authentic." These subfolders categorize the logo images
based on their authenticity, with the "fake" directory containing images of counterfeit logos
and the "authentic" directory housing images of genuine logos. Each subfolder comprises a
collection of logo images, directly stored without further subdivision. Each image within
these subfolders is labeled to indicate whether it represents a genuine or fake logo. This
organized structure simplifies data access and management, making it easier to identify the
data and develop models for fake logo detection.


**MODELLING**
**Model Development**
We have developed a VGG16 model for fake logo detection. This model is specifically
designed to analyze logo images and classify them as either genuine or fake. Its purpose is to
contribute to the broader goal of enabling businesses and consumers to better protect
themselves against fraudulent practices, maintain brand trust, and make informed purchasing
decisions.

**VGG16 Model**
**Model Building**
# Flattening the output layer of VGG16
x = Flatten()(vgg16.output)
# Adding a dense layer for predictions
prediction = Dense(len(folders), activation='sigmoid')(x)
# Creating the model
model = Model(inputs=vgg16.input, outputs=prediction)
# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

We flatten the output layer of the VGG16 model to prepare it for the fully connected layers
then, we add a dense layer for predictions, with the number of units equal to the number of
output classes (the length of the `folders` list) and using the sigmoid activation function. Next,
we create the model using the VGG16 input and the prediction output.Finally, we compile the
model using categorical crossentropy as the loss function, the Adam optimizer, and accuracy
as the metric to monitor during training.

**Image Data Augmentation and Loading:**
# Data Augmentation for training set
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2,
horizontal_flip=True)
# Data Augmentation for test set
test_datagen = ImageDataGenerator(rescale=1./255)
# Loading Training and Test Data
training_set = train_datagen.flow_from_directory(train_path, target_size=(224, 224),
batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory(valid_path, target_size=(224, 224),
batch_size=32, class_mode='categorical')

We define two `ImageDataGenerator` objects for data augmentation, one for the training set
and one for the test set.For the training set, we apply rescaling, shearing, zooming, and
horizontal flipping transformations to augment the data. For the test set, we only apply
rescaling to keep the data consistent.We then use the `flow_from_directory` method to load
the training and test data from the specified directories, with a target size of (224, 224) and a
batch size of 32, while also specifying that the class mode is categorical.

**Model Training:**
# Training the model
r = model.fit_generator(training_set, validation_data=test_set, epochs=50,
steps_per_epoch=len(training_set), validation_steps=len(test_set))

We train the model using the `fit_generator` method.We provide the training set as the training
data and the test set as the validation data.We specify the number of epochs as 50.The
`steps_per_epoch` parameter is set to the number of batches in the training set, and
`validation_steps` is set to the number of batches in the test set.

**Model Visualization:**
# Plotting the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
# Plotting the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

We plot the training and validation loss values over the epochs to visualize the model's
performance.Similarly, we plot the training and validation accuracy values to analyze how well
the model is learning from the data.

**Saving the model:**
# Saving the trained model
model.save('/content/drive/MyDrive/model2_vgg16.h5')
This code snippet saves the trained model to a file named "model2_vgg16.h5" in the
specified directory on Google Drive ("/content/drive/MyDrive/").

**Building an Interface**
We built an interface using Flask. This Flask web application serves as a tool for detecting
fake logos in uploaded images. It utilizes a pre-trained VGG16 model to analyze the
uploaded logo images and classify them as either fake or original. Upon receiving an image
through the web interface, the application preprocesses the image, makes predictions using
the loaded model, and interprets the results. The prediction, along with the uploaded image, is
then displayed on the result page. Users can upload an image of a logo, and the application
promptly determines whether it's genuine or counterfeit.

**Sample Code**
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import tensorflow as tf
print(tf.__version__)
# import the libraries as shown below
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
# re-size all the images to this
IMAGE_SIZE = [224, 224]
train_path = '/content/drive/MyDrive/Images splitted/train'
valid_path = '/content/drive/MyDrive/Images splitted/val'
# Import the VGG16 library as shown below and add preprocessing layer to the front of
VGG
# Here we will be using imagenet weights
vgg16 = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet',
include_top=False)
# don't train existing weights
for layer in vgg16.layers:
layer.trainable = False
# useful for getting number of output classes
folders = glob('/content/drive/MyDrive/Images/*')
folders
# our layers - you can add more if you want
x = Flatten()(vgg16.output)
len(folders)
prediction = Dense(len(folders), activation='sigmoid')(x)
# create a model object
model = Model(inputs=vgg16.input, outputs=prediction)
# view the structure of the model
model.summary()
# tell the model what cost and optimization method to use
model.compile(
loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy']
)
# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/Images
splitted/train',
target_size = (224, 224),
batch_size = 32,
class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/Images splitted/val',
target_size = (224, 224),
batch_size = 32,
class_mode = 'categorical')
# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
training_set,
validation_data=test_set,
epochs=50,
steps_per_epoch=len(training_set),
validation_steps=len(test_set)
)
# Evaluate the model on the test set
eval_result = model.evaluate(test_set, steps=len(test_set))
# Calculate the accuracy
total_accuracy = eval_result[1] # Assuming the accuracy metric is the second one returned
by evaluate()
print("Total Accuracy:", total_accuracy)
import matplotlib.pyplot as plt
# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')
# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
# save it as a h5 file
from tensorflow.keras.models import load_model
model.save('/content/drive/MyDrive/model2_vgg16.h5')
y_pred = model.predict(test_set)
y_pred
import numpy as np
y_pred = np.argmax(y_pred, axis=1)
y_pred
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model('/content/drive/MyDrive/model2_vgg16.h5')
img=image.load_img('/content/drive/MyDrive/Images/Fake/112.jpg',target_size=(224,224)
)
x=image.img_to_array(img)
x
x.shape
x=x/255
import matplotlib.pyplot as plt
x = x.reshape((-1, 224, 224, 3))
predictions = model.predict(x)
# Interpret the predictions
print(predictions)
fake_probability = predictions[0][0] # Assuming first index represents the probability of
being fake
# Display the image with the prediction
plt.imshow(img)
if predictions[0][0] > predictions[0][1]:
plt.title(f"Fake (Probability: {fake_probability:.2f})")
else:
plt.title(f"Real (Probability: {1 - fake_probability:.2f})")
plt.axis('off')
plt.show()
