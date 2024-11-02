# %%
#Source of Base Tensorflow model
import tensorflow as tf
import os
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import matplotlib.pyplot as plt
import re
import splitfolders
import shutil
import random
import tarfile

# %%
#Unzip part1.tar.gz (from UTKFaces here: )
#Create two folders for sorting images into, #male and #female
tar_gz_file_path = 'part1.tar.gz'
try:
     with tarfile.open(tar_gz_file_path, 'r:gz') as tar_ref:
      tar_ref.extractall()
      untar_dir = os.path.splitext(os.path.splitext(tar_gz_file_path)[0])[0]
      os.makedirs(os.path.join(untar_dir, 'male'), exist_ok=True)
      os.makedirs(os.path.join(untar_dir, 'female'), exist_ok=True)
except Exception as e:
    print(f"An error occurred: {e}")

# %%
# The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg

# [age] is an integer from 0 to 116, indicating the age
# [gender] is either 0 (male) or 1 (female)
# [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
# [date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace

#Move files into their own files divided by gnder, then create test and validation folders
def move_files_by_name(source_dir, destination_dir, pattern):
    for filename in os.listdir(source_dir):
        if re.match(pattern, filename):
            source_path = os.path.join(source_dir, filename)
            destination_path = os.path.join(destination_dir, filename)
            shutil.move(source_path, destination_path)
move_files_by_name('part1','part1/male',r"\d+_0.*")
move_files_by_name('part1','part1/female',r"\d+_1.*")


#Split into train and test folders
splitfolders.ratio('part1', output="output", seed=1337, ratio=(.8,.2)) 

# %%
train_data_dir = 'output/train'
test_data_dir = 'output/val'

class_names = sorted(os.listdir(train_data_dir)) 

def load_images_from_folder(folder_path):
    images = []
    labels = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(folder_path, class_name)
        if os.path.isdir(class_dir):
            for image_file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_file)
                try:
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
                    image = preprocess_input(image)
                    images.append(image)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    return np.array(images), np.array(labels)

#Set test and train files
train_images, train_labels = load_images_from_folder(train_data_dir)
test_images, test_labels = load_images_from_folder(test_data_dir)

# %%
test_images.shape

# %%
#Plot Example Images to ensure everything looks correct
class_names = ['female','male']
random_indices = random.sample(range(len(train_images)), 25) 

plt.figure(figsize=(10,10))
for i, idx in enumerate(random_indices):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[idx])  # Use the random index
    plt.xlabel(class_names[train_labels[idx]])  # Use the random index
plt.show()

# %%
#Set Base Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the first few layers
for layer in base_model.layers[:150]:  # Experiment with the number of layers to freeze
    layer.trainable = False

# Add layers
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(5, kernel_regularizer='l2')  # Add L2 regularization
softmax_layer = tf.keras.layers.Softmax()

# Create the complete model
model = tf.keras.Sequential(
    [base_model, global_average_layer, prediction_layer, softmax_layer]
)

# %%
#Compile Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=7, 
                    validation_data=(test_images, test_labels))

# %%
#Plot Accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# %%
#Save the model for use in Ace algorithms
model.save("models/gender_classification.h5")

#Get summary so I can feed in bottleneck layer and input layers into custom_wrapper later
model.summary()

# %%



