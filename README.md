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
